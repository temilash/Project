import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import torchaudio.transforms as T


EPS = 1e-8

class ConvTasNetStereo(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        n_fft=512,
        hop_length=256,
        input_channels=2,
        frame_length=64,
        frame_step=32,
        samplerate=44100,
    ):
        """
        Args:
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            input_channels: Number of input channels (mono/stereo)
            frame_length: Length of chunks for chunked separation
            frame_step: Step size for overlapping chunks
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.samplerate = samplerate
        # Components
        self.encoder = SpectrogramEncoder(n_fft=n_fft, hop_length=hop_length)
        self.separator = CausalHybridDPTSeparator(input_channels=256)
        self.decoder = SpectrogramDecoder(n_fft=n_fft, hop_length=hop_length)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        return length

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        spectrogram, phase = self.encoder(mixture)
        
        # Step 2: Chunking for Overlap-and-Add
        spectrogram_chunks = self.chunk_input(spectrogram)

        # Step 3: Process each chunk independently
        separated_chunks = []
        for chunk in spectrogram_chunks:
            mask = self.separator(chunk)
            mask = mask[:, :, :, :chunk.size(-1)]  # Ensure mask matches chunk size
            separated_chunk = chunk * mask  # Apply the mask
            separated_chunks.append(separated_chunk)

        # Step 4: Merge chunks using Overlap-and-Add
        separated_spectrogram = self.overlap_add(torch.stack(separated_chunks), self.frame_step)
        

        separated_spectrogram = self.decoder(separated_spectrogram, phase)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)  # Original waveform length
        T_conv = separated_spectrogram.size(-1)  # Decoded waveform length after ISTFT
        separated_spectrogram = F.pad(separated_spectrogram, (0, T_origin - T_conv))  # Fix length mismatch

        return separated_spectrogram

    def serialize(self):
        """Serialize model and output dictionary.

        Returns:
            dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            # model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__,
            pytorch_lightning_version=pl.__version__,
            asteroid_version="0.7.0",
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        fb_config = self.encoder.filterbank.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is"
                " not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args
        
    def chunk_input(self, spectrogram):
        """
        Splits the spectrogram into overlapping chunks.
        Args:
            spectrogram: [batch, channels, freq_bins, time]
        Returns:
            List of chunks: [chunk1, chunk2, ...]
        """
        batch, channels, freq_bins, time = spectrogram.shape
        chunks = []

        for start in range(0, time - self.frame_length + 1, self.frame_step):
            chunk = spectrogram[:, :, :, start:start + self.frame_length]
            chunks.append(chunk)

        return chunks  # List of tensors (each is [batch, channels, freq_bins, frame_length])

    def overlap_add(self, chunks, frame_step):
        """
        Merges overlapping chunks using Overlap-and-Add.
        Args:
            chunks: [num_chunks, batch, channels, freq_bins, frame_length]
            frame_step: Overlap step size
        Returns:
            Merged spectrogram: [batch, channels, freq_bins, time]
        """
        num_chunks, batch, channels, freq_bins, frame_length = chunks.shape
        output_size = frame_step * (num_chunks - 1) + frame_length
        output = torch.zeros(batch, channels, freq_bins, output_size, device=chunks.device)

        for i, chunk in enumerate(chunks):
            output[:, :, :, i * frame_step: i * frame_step + frame_length] += chunk

        return output


class SpectrogramEncoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1)  # Power=1 keeps phase info
        self.conv2d_freq = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(5,5), padding=(2,2))

    def forward(self, waveform):
        """
        Args:
            waveform: [batch, channels, time]
        Returns:
            encoded_spectrogram: [batch, 256, time_frames]
        """
        print(f"Input Mixture Shape: {waveform.shape}")
        spectrogram = self.stft(waveform)  # [batch, 2, freq_bins, time_frames] (Stereo)
        phase = torch.angle(spectrogram)  # âœ… Extract phase for both channels

        spectrogram = spectrogram.abs()  # Take magnitude
        spectrogram = self.conv2d_freq(spectrogram)  # Apply 2D CNN
        
        print(f" Encoder Output Shapes:")
        print(f"  - Spectrogram shape: {spectrogram.shape}")  # Should be [batch, 256, freq_bins, time_frames]
        print(f"  - Phase shape: {phase.shape}")  # Should be [batch, 2, freq_bins, time_frames]


        return spectrogram, phase

class SpectrogramDecoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.deconv2d = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=2,  # Match the number of channels in phase
            kernel_size=(5, 5),
            padding=(2, 2)
        )

    def forward(self, encoded_spec, phase, mixture):
        """
        Args:
            encoded_spec: [batch, 256, freq_bins, time]  (Processed spectrogram from 2D CNN)
            phase: [batch, 2, freq_bins, time]  (Saved phase from encoder)
            mixture: [batch, 2, time]  (Original input waveform for length reference)
        Returns:
            waveform: [batch, 2, time]  (Reconstructed waveform matching mixture shape)
        """

        print("\n=== Decoder Debugging ===")
        print(f"Encoded Spectrogram Shape: {encoded_spec.shape}")  
        print(f"Phase Shape: {phase.shape}")  

        # Convert from 256 channels to 2 channels
        mag_est = self.deconv2d(encoded_spec)  
        print(f"After Deconv2D Shape: {mag_est.shape}")  

        _, _, freq_bins, time_frames = phase.shape  # Get correct frequency-time dimensions

        # Ensure mag_est and phase match in time dimension
        _, _, _, time_mag = mag_est.shape

        if time_mag > time_frames:
            print(f"Cropping mag_est from {time_mag} to {time_frames}")
            mag_est = mag_est[:, :, :, :time_frames]
        elif time_mag < time_frames:
            pad_amount = time_frames - time_mag
            print(f"Padding mag_est by {pad_amount} to match phase")
            mag_est = F.pad(mag_est, (0, pad_amount))

        print(f"Final mag_est Shape Before ISTFT: {mag_est.shape}")

        # Ensure non-negative magnitude
        mag_est = mag_est.clamp(min=0)

        # Convert magnitude + phase back to complex form
        complex_spec = torch.polar(mag_est, phase)  
        print(f"Complex Spectrogram Shape: {complex_spec.shape}")

        batch, channels, freq_bins, time_frames = complex_spec.shape

        # í ½í´¥ **FIX: Multiply freq_bins * time_frames to match the time dimension**
        complex_spec = complex_spec.view(batch, channels, freq_bins * time_frames)  

        print(f"Reshaped for ISTFT (Fixed): {complex_spec.shape}")

        # Apply ISTFT
        waveform = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=None
        )

        print(f"ISTFT Output Shape (Before Fixing Dimensions): {waveform.shape}")

        # í ½í´¥ **Ensure final output shape matches mixture**
        T_target = mixture.size(-1)  # 661500
        T_decoded = waveform.size(-1)  

        if T_decoded > T_target:
            print(f"Trimming output from {T_decoded} to {T_target}")
            waveform = waveform[:, :, :T_target]  
        elif T_decoded < T_target:
            print(f"Padding output by {T_target - T_decoded} samples")
            waveform = F.pad(waveform, (0, T_target - T_decoded))  

        print(f"Final Matched Output Shape: {waveform.shape}")  # Should be [batch, 2, 661500]

        return waveform




"""def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes, device=signal.device).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result"""

class CausalConv2D(nn.Module):
    """
    Implements a causal 2D convolution where the time dimension only depends on past values.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3), dilation=(1, 1)):
        super().__init__()
        padding_time = (kernel_size[1] - 1) * dilation[1]
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size[0] // 2, padding_time)  # Asymmetric padding in time

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size[0] // 2, 0),  # Ensure no change in time dimension
            dilation=dilation
        )

    def forward(self, x):
        return F.relu(self.conv(x))

class GatedCausalConv2D(nn.Module):
    """
    Gated causal 2D convolution for improved temporal modeling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3), dilation=(1, 1)):
        super().__init__()
        self.conv_f = CausalConv2D(in_channels, out_channels, kernel_size, dilation)
        self.conv_g = CausalConv2D(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x):
        return torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))

class CausalHybridDPTSeparator(nn.Module):
    """
    Fully 2D CNN-based causal separator without reshaping the spectrogram.
    """

    def __init__(self, input_channels=256, tcn_depth=4):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            CausalConv2D(input_channels, 128, kernel_size=(5, 1)),  # No time change
            CausalConv2D(128, 128, kernel_size=(5, 1))  # No time change
        )

        # Ensure dilated convolutions do NOT shrink time dimension
        self.tcn_blocks = nn.Sequential(
            *[
                nn.Sequential(
                    CausalConv2D(128, 128, kernel_size=(5, 1), dilation=(1, 2**i)),  # No time change
                    CumulativeLayerNorm(128)  # Normalization
                )
                for i in range(tcn_depth)
            ]
        )

        self.gated_conv = GatedCausalConv2D(128, 128, kernel_size=(5, 1))

        # Final convolution that ensures time dimension is unchanged
        self.final_conv = nn.Conv2d(128, input_channels, kernel_size=1, padding=0)

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: [batch, channels, freq_bins, time]  (No reshaping)
        Returns:
            mask: [batch, channels, freq_bins, time] - Mask for separation.
        """

        # Step 1: Extract features using causal 2D convolutions
        spectrogram = self.feature_extractor(spectrogram)

        # Step 2: Apply deep causal TCN layers
        spectrogram = self.tcn_blocks(spectrogram)

        # Step 3: Apply gated convolution
        spectrogram = self.gated_conv(spectrogram)

        # Step 4: Predict final mask
        mask = self.final_conv(spectrogram)

        # Ensure the mask is between 0 and 1
        mask = torch.sigmoid(mask)

        return mask



class CumulativeLayerNorm(nn.Module):
    """Cumulative Layer Normalization (CLN)"""

    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, C, 1, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, C, 1, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [batch, channels, freq_bins, time]
        Returns:
            cln_y: Cumulative Layer Normalized tensor
        """
        batch, channels, freq_bins, time = y.shape

        # Compute cumulative mean & variance along the **time** axis
        cumulative_mean = y.cumsum(dim=-1) / (torch.arange(1, time + 1, device=y.device).view(1, 1, 1, -1))
        cumulative_var = ((y - cumulative_mean) ** 2).cumsum(dim=-1) / (torch.arange(1, time + 1, device=y.device).view(1, 1, 1, -1))

        # Normalize using cumulative statistics
        cln_y = self.gamma * (y - cumulative_mean) / torch.sqrt(cumulative_var + 1e-5) + self.beta
        return cln_y


if __name__ == "__main__":
    torch.manual_seed(123)
    M, N, L, T = 2, 3, 4, 12
    K = 2 * T // L - 1
    B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 2, "gLN", False
    mixture = torch.randint(3, (M, T))
    # test Encoder
    encoder = Encoder(L, N, 1)
    encoder.conv1d_U.weight.data = torch.randint(2, encoder.conv1d_U.weight.size())
    mixture_w = encoder(mixture)
    print("mixture", mixture)
    print("U", encoder.conv1d_U.weight)
    print("mixture_w", mixture_w)
    print("mixture_w size", mixture_w.size())

    # test TemporalConvNet
    separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
    est_mask = separator(mixture_w)
    print("est_mask", est_mask)

    # test Decoder
    decoder = Decoder(N, L, audio_channels=1)
    est_mask = torch.randint(2, (B, K, C, N))
    est_source = decoder(mixture_w, est_mask)
    print("est_source", est_source)

    # test Conv-TasNet
    conv_tasnet = ConvTasNetStereo(N, L, B, H, P, X, R, C, norm_type=norm_type)
    est_source = conv_tasnet(mixture)
    print("est_source", est_source)
    print("est_source size", est_source.size())
