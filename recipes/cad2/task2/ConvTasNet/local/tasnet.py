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
        num_sources=2,
        num_repeats=4,
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
        self.input_channels = input_channels
        self.num_sources = num_sources
        self.num_repeats = num_repeats
    
        # Components
        self.encoder = SpectrogramEncoder(n_fft=n_fft, hop_length=hop_length)
        self.separator = CausalHybridGRUSeparator(
            input_channels=256,  # Encoder outputs 256
            num_sources=self.num_sources,
            num_repeats=self.num_repeats
        )
        self.decoder = SpectrogramDecoder(n_fft=n_fft, hop_length=hop_length)
    
        # Weight init
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
        spectrogram,phase = self.encoder(mixture)
        
        masked_chunks = self.separator(spectrogram)
        
        #masked_spec = self.overlap_add(masked_chunks, self.frame_step)
        
        input_length = mixture.size(-1)
        separated_waveforms = self.decoder(masked_chunks, phase,input_length)
        
        print(f"waveform shape {separated_waveforms.shape}")

        return separated_waveforms

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

    def overlap_and_add(chunks, frame_step):
        """
        Reconstructs full spectrogram from overlapped chunks using advanced overlap-add.
    
        Args:
            chunks: [N, B, C, S, F, K]
            frame_step: step size between chunks (hop length)
        
        Returns:
            full_spec: [B, C, S, F, T]
        """
        N, B, C, S, F, K = chunks.shape
    
        outer_dims = (B, C, S, F)
        total_frames = frame_step * (N - 1) + K
    
        subframe_len = math.gcd(K, frame_step)
        subframes_per_chunk = K // subframe_len
        subframe_step = frame_step // subframe_len
        total_subframes = total_frames // subframe_len
    
        # Reshape chunks to subframes
        subframe_chunks = chunks.reshape(N, *outer_dims, subframes_per_chunk, subframe_len)
        subframe_chunks = subframe_chunks.permute(1, 2, 3, 4, 0, 5, 6)  # [B, C, S, F, N, subframes, subframe_len]
        subframe_chunks = subframe_chunks.reshape(*outer_dims, -1, subframe_len)  # [B, C, S, F, N*subframes, subframe_len]
    
        # Create unfolding indices
        frame = torch.arange(0, total_subframes, device=chunks.device).unfold(
            0, subframes_per_chunk, subframe_step
        )
        frame = frame.contiguous().view(-1)[:subframe_chunks.shape[-2]]  # [N * subframes]
        
        # Add overlapping subframes
        output = torch.zeros(*outer_dims, total_subframes, subframe_len, device=chunks.device)
        output.index_add_(-2, frame, subframe_chunks)
    
        # Reshape back to [B, C, S, F, T]
        full_spec = output.reshape(*outer_dims, -1)
    
        return full_spec


class SpectrogramEncoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.hop_length = hop_length
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
        phase = torch.angle(spectrogram)  # ✅ Extract phase for both channels

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

        # Reduce 256 → 2 channels (stereo)
        self.reduce_conv = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, masked_spec, phase, input_length):
        """
        Args:
            masked_spec: [B, C, S, freq, T_spec]  (Separated magnitude spectrogram)
            phase: [B, 2, freq, T_phase] — shared phase for all sources
        Returns:
            waveform: [B, S, 2, time]
        """
        B, C, S, freq, T_spec = masked_spec.shape
        T_phase = phase.size(-1)

        # If masked_spec has a shorter time dimension than phase, pad it.
        if T_spec < T_phase:
            diff = T_phase - T_spec
            # Pad along the time dimension (last dimension) on the right.
            masked_spec = F.pad(masked_spec, (0, diff))
        
        waveforms = []

        for s in range(S):
            spec_s = masked_spec[:, :, s, :, :]          # [B, C, freq, T_phase]
            spec_s = self.reduce_conv(spec_s)            # [B, 2, freq, T_phase]
            # Use clamped magnitude and full phase to create a complex spectrogram
            complex_spec = torch.polar(spec_s.clamp(min=0), phase)  # [B, 2, freq, T_phase]

            # Reshape for ISTFT: combine batch and channel dims
            b, ch, f, t = complex_spec.shape
            complex_spec = complex_spec.view(b * ch, f, t)
            
            print(f"  - Spectrogram shape: {complex_spec.shape}") 

            # Inverse STFT to recover time-domain waveform
            waveform = torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=None, length=input_length)
            
            print(f"  - Spectrogram shape: {waveform.shape}") 

            # Reshape back to [B, 2, time]
            waveform = waveform.view(b, ch, -1)
            waveforms.append(waveform)  # [B, 2, time]
            
        print(f"  - Spectrogram shape: {waveform.shape}") 

        return torch.stack(waveforms, dim=1)  # [B, S, 2, time]


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
    Implements causal 2D convolution: future time steps are not accessed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=(5, 3), dilation=(1, 1)):
        super().__init__()
        pad_freq = kernel_size[0] // 2  # symmetric freq padding is fine
        pad_time = (kernel_size[1] - 1) * dilation[1]

        self.pad_time = pad_time
        self.pad = (0, pad_time)  # right-pad time only (causal)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(pad_freq, 0),  # freq is symmetric
            dilation=dilation
        )

    def forward(self, x):
        # Causal padding in time dimension (only right side)
        x = F.pad(x, self.pad)  # [left, right] on time dim only
        out = self.conv(x)

        # Remove extra future-padded steps (causal chomp)
        if self.pad_time > 0:
            out = out[..., :-self.pad_time]  # remove future steps
        return F.relu(out)

class CausalHybridGRUSeparator(nn.Module):
    def __init__(self, input_channels=256, hidden_channels=128, num_sources=2, num_repeats=4):
        super().__init__()
        self.num_sources = num_sources
        self.num_repeats = num_repeats

        # Short-term modeling with causal convolutions
        self.causal_conv1 = CausalConv2D(input_channels, hidden_channels, kernel_size=(5, 3), dilation=(1, 1))
        self.causal_conv2 = CausalConv2D(hidden_channels, hidden_channels, kernel_size=(5, 3), dilation=(1, 2))

        # Long-term modeling: create a list of GRU layers, each applied sequentially.
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_channels,
                hidden_size=hidden_channels,
                num_layers=2,  # Single-layer GRU per repetition
                batch_first=True,
                bidirectional=False
            )
            for _ in range(num_repeats)
        ])

        # Final projection to mask space: project hidden features to (input_channels * num_sources)
        self.mask_proj = nn.Conv2d(hidden_channels, input_channels * num_sources, kernel_size=1)

    def forward(self, spectrogram):
        """
        Args:
            spectrogram: [B, C, freq, T_in]  (Full spectrogram)
        Returns:
            masked_spec: [B, C, num_sources, freq, T_new]  (Masked spectrogram for each source)
        """
        B, C, freq, T_in = spectrogram.shape

        # Step 1: Apply causal convolutions for short-term modeling
        out = self.causal_conv1(spectrogram)  # [B, hidden_channels, freq, T_conv]
        out = self.causal_conv2(out)          # [B, hidden_channels, freq, T_conv]
        H = out.size(1)
        print(f"out shape1 = {out.shape}")

        # Step 2: Process the full time sequence with the GRU layers for long-term dependencies.
        # Permute so GRU processes along the time dimension: [B, freq, H, T_conv]
        out = out.permute(0, 2, 1, 3).contiguous()  
        print(f"out shape2 = {out.shape}")
        T_conv = out.size(-1)
        # Reshape to combine batch and frequency dimensions: [B*freq, T_conv, H]
        out = out.view(B * freq, H, T_conv).transpose(1, 2)
        print(f"out shape 3 = {out.shape}")

        # Apply GRU repeatedly over the time dimension
        for gru in self.gru_layers:
            out, _ = gru(out)  # [B*freq, T_new, H]

        # Capture new time dimension after GRU processing
        T_new = out.size(1)
        print(f"Time dimension after GRU = {T_new}")

        # Reshape back: [B, freq, H, T_new] and then permute to [B, H, freq, T_new]
        out = out.transpose(1, 2).contiguous().view(B, freq, H, T_new).permute(0, 2, 1, 3)
        print(f"out shape after GRU and reshape = {out.shape}")

        # Step 3: Project features to mask space
        mask = self.mask_proj(out)  # [B, (input_channels*num_sources), freq, T_new]
        # Reshape to explicitly include the source dimension: [B, num_sources, input_channels, freq, T_new]
        mask = mask.view(B, self.num_sources, C, freq, T_new)
        mask = F.relu(mask)  # Ensure non-negative mask values
        print(f"mask shape = {mask.shape}")

        # Step 4: Apply masks to the original spectrogram.
        # If needed, trim the original spectrogram to T_new
        if T_in != T_new:
            spectrogram = spectrogram[..., :T_new]
        # Expand original spectrogram to include a source dimension: [B, 1, C, freq, T_new]
        spec_expanded = spectrogram.unsqueeze(1)
        # Element-wise multiplication: [B, num_sources, C, freq, T_new]
        masked_spec = spec_expanded * mask
        # Permute to get final output: [B, C, num_sources, freq, T_new]
        masked_spec = masked_spec.permute(0, 2, 1, 3, 4)

        return masked_spec



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
