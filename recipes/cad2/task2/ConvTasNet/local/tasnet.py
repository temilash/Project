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
        frame_length=128,
        frame_step=64,
        samplerate=44100,
        num_sources=2,
        num_repeats=4,
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.samplerate = samplerate
        self.input_channels = input_channels
        self.num_sources = num_sources
        self.num_repeats = num_repeats
    
        self.encoder = SpectrogramEncoder(n_fft=n_fft, hop_length=hop_length)
        self.separator = CausalDualPathRNNSeparator(
            input_channels=256,
            hidden_channels=128,
            num_sources=self.num_sources,
            num_blocks=self.num_repeats,  # Number of DPRNN blocks
            chunk_size=self.frame_length,  # Size of each chunk
        )
        self.decoder = SpectrogramDecoder(n_fft=n_fft, hop_length=hop_length)
    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        return length

    def forward(self, mixture):
        spectrogram = self.encoder(mixture)
        masked_chunks = self.separator(spectrogram)
        input_length = mixture.size(-1)
        separated_waveforms = self.decoder(masked_chunks,mixture ,input_length)
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

class SpectrogramEncoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Compute a complex spectrogram (power=None returns complex)
        self.stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None, normalized=True)

        # Now expecting 6-channel input: [log_mag_L, sin_L, cos_L, log_mag_R, sin_R, cos_R]
        self.conv2d = nn.Conv2d(in_channels=6, out_channels=256, 
                                kernel_size=(5, 5), padding=(2, 2))

    def forward(self, waveform):
        """
        Args:
            waveform: Tensor of shape [B, 2, T]  # stereo input
        Returns:
            encoded_spec: [B, 128, F, T']
            phase_L: [B, F, T']
            phase_R: [B, F, T']
        """
        # Compute STFT on both channels independently
        complex_spec = self.stft(waveform)  # [B, 2, F, T']

        # Split left and right channels
        complex_L = complex_spec[:, 0]  # [B, F, T']
        complex_R = complex_spec[:, 1]  # [B, F, T']

        # Compute mag & phase for each channel
        mag_L = torch.abs(complex_L).unsqueeze(1)  # [B, 1, F, T']
        mag_R = torch.abs(complex_R).unsqueeze(1)
        log_mag_L = torch.log1p(mag_L)
        log_mag_R = torch.log1p(mag_R)

        phase_L = torch.angle(complex_L)
        phase_R = torch.angle(complex_R)

        sin_L = torch.sin(phase_L).unsqueeze(1)
        cos_L = torch.cos(phase_L).unsqueeze(1)
        sin_R = torch.sin(phase_R).unsqueeze(1)
        cos_R = torch.cos(phase_R).unsqueeze(1)

        # Concatenate 6 channels: [B, 6, F, T']
        features = torch.cat([log_mag_L, sin_L, cos_L, log_mag_R, sin_R, cos_R], dim=1)

        # Pass through convolutional encoder
        encoded_spec = self.conv2d(features)  # [B, 128, F, T']

        return encoded_spec


class SpectrogramDecoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, num_sources=2, num_channels=2):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_sources = num_sources
        self.num_channels = num_channels
        
        self.register_buffer(
            "window", 
            torch.hann_window(n_fft, periodic=True),
            persistent=False
        )

    def forward(self, mask, mixture, input_length):
        """
        Args:
            mask: Tensor [B, S, C, 2, F, T] - Complex ratio mask from separator
                 where 2 = {real_mask, imag_mask}
            mixture: Tensor [B, C, time] - Original stereo mixture
            input_length: int - Original length of input
        Returns:
            separated: Tensor [B, S, C, time] - Separated sources
        """
        B, S, C, _, F, T = mask.shape
        
        # 1. Compute STFT of mixture for each channel
        mix_stft_list = []
        for c in range(C):
            stft = torch.stft(
                mixture[:, c],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                normalized=True,
                return_complex=True,
                center=True
            )  # [B, F, T']
            mix_stft_list.append(stft)
        
        # Stack channels
        mix_stft = torch.stack(mix_stft_list, dim=1)  # [B, C, F, T']
        
        # 2. Apply complex ratio mask for each source
        separated = []
        for s in range(self.num_sources):
            # Get real and imaginary parts of the mask
            M_r = mask[:, s, :, 0]  # Real part [B, C, F, T]
            M_i = mask[:, s, :, 1]  # Imaginary part [B, C, F, T]
            
            # Complex multiplication: (a + bi)(c + di) = (ac-bd) + (ad+bc)i
            X_r = mix_stft.real
            X_i = mix_stft.imag
            
            # Compute separated spectrogram
            Y_r = M_r * X_r - M_i * X_i  # Real part
            Y_i = M_r * X_i + M_i * X_r  # Imaginary part
            
            # Combine to complex
            Y = torch.complex(Y_r, Y_i)  # [B, C, F, T]
            
            # ISTFT for each channel
            source_channels = []
            for c in range(C):
                wav = torch.istft(
                    Y[:, c],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    normalized=True,
                    length=input_length
                )  # [B, time]
                source_channels.append(wav)
            
            # Stack channels for this source
            source = torch.stack(source_channels, dim=1)  # [B, C, time]
            separated.append(source)
        
        # Stack all sources
        separated = torch.stack(separated, dim=1)  # [B, S, C, time]
        return separated

####################################################
# I changed the cum layer norm run again with this
######################################################

class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size))
        self.beta = nn.Parameter(torch.zeros(channel_size))
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T] or [B, T, C] where:
                B: batch size
                C: channels
                T: time steps
        """
        # Ensure channel dimension is in the middle
        if x.size(1) != self.channel_size:
            x = x.transpose(1, 2)
            
        # Compute statistics over the channel dimension
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, T]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [B, 1, T]
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + EPS)
        
        # Apply affine transformation
        gamma = self.gamma.view(1, -1, 1)  # [1, C, 1]
        beta = self.beta.view(1, -1, 1)    # [1, C, 1]
        output = gamma * x_norm + beta
        
        # Return to original dimension ordering if needed
        if x.size(1) != self.channel_size:
            output = output.transpose(1, 2)
            
        return output

class CausalDualPathRNNSeparator(nn.Module):
    def __init__(self, input_channels=256, hidden_channels=128, num_sources=2,
                 num_blocks=6, chunk_size=100):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_sources = num_sources
        self.chunk_size = chunk_size
        
        # Input transformation
        self.norm = ChannelwiseLayerNorm(input_channels)
        self.in_proj = nn.Linear(input_channels, hidden_channels)
        
        # DPRNN blocks
        self.dprnn_blocks = nn.ModuleList([
            CausalDualPathRNNBlock(hidden_channels) for _ in range(num_blocks)
        ])
        
        # Output projection for complex masks (2 channels * 2 real/imag)
        self.out_proj = nn.Linear(hidden_channels, num_sources * 2 * 2)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, F, T] - Input spectrogram
        Returns:
            complex_mask: [B, S, C, 2, F, T]
        """
        B, C, freq, T = x.shape
        
        # Reshape and normalize
        x = x.permute(0, 2, 1, 3)  # [B, F, C, T]
        x = x.reshape(B*freq, C, T)
        x = self.norm(x)
        x = x.reshape(B, freq, C, T)
        
        # Create chunks
        num_chunks = math.ceil(T / self.chunk_size)
        pad_length = num_chunks * self.chunk_size - T
        if pad_length > 0:
            x = F.pad(x, (0, pad_length))
            
        # Process chunks
        x = x.reshape(B, freq, C, num_chunks, self.chunk_size)
        x = x.permute(0, 1, 3, 4, 2)  # [B, F, chunks, chunk_size, C]
        x = x.reshape(B*freq, num_chunks, self.chunk_size, C)
        x = self.in_proj(x)
        
        # DPRNN processing
        hidden_states = None
        for block in self.dprnn_blocks:
            x, hidden_states = block(x, hidden_states)
            
        # Generate masks
        output = self.out_proj(x)  # [B*F, chunks, chunk_size, S*2*2]
        
        # Reshape to separate sources, channels and real/imag components
        output = output.view(B, freq, num_chunks, self.chunk_size, 
                           self.num_sources, 2, 2)  # [B, F, chunks, chunk_size, S, C, 2]
        
        # Rearrange dimensions
        output = output.permute(0, 4, 5, 6, 1, 2, 3)  # [B, S, C, 2, F, chunks, chunk_size]
        output = output.reshape(B, self.num_sources, 2, 2, freq, -1)
        
        # Trim padding
        if pad_length > 0:
            output = output[..., :T]
            
        # Apply sigmoid for mask values
        complex_mask = torch.sigmoid(output)
        
        return complex_mask
class CausalDualPathRNNBlock(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        self.intra_gru = nn.GRU(
            hidden_channels, 
            hidden_channels,
            batch_first=True,
            bidirectional=False
        )
        
        self.inter_gru = nn.GRU(
            hidden_channels, 
            hidden_channels,
            batch_first=True,
            bidirectional=False
        )
        
        self.intra_norm = ChannelwiseLayerNorm(hidden_channels)
        self.inter_norm = ChannelwiseLayerNorm(hidden_channels)
        
        # Add batch size tracking
        self.last_batch_size = None

    def forward(self, x, hidden_states=None):
        B, num_chunks, chunk_size, H = x.shape
        #assert H == self.hidden_channels, f"Expected hidden size {self.hidden_channels}, got {H}"
        
        # Reset hidden states if batch size changes
        if self.last_batch_size != B*num_chunks:
            hidden_states = None
            self.last_batch_size = B*num_chunks
        
        if hidden_states is None:
            # Initialize hidden states with correct dimensions
            h_intra = torch.zeros(1, B*num_chunks, H, device=x.device)
            h_inter = torch.zeros(1, B*chunk_size, H, device=x.device)
            hidden_states = (h_intra, h_inter)
        else:
            h_intra, h_inter = hidden_states
            
        # Intra-chunk processing
        intra_input = x.reshape(B*num_chunks, chunk_size, H)
        intra_output, h_intra = self.intra_gru(intra_input, h_intra)
        intra_output = intra_output.transpose(1, 2)
        intra_output = self.intra_norm(intra_output)
        intra_output = intra_output.transpose(1, 2)
        intra_output = intra_output.reshape(B, num_chunks, chunk_size, H)
        x = x + intra_output
        
        # Inter-chunk processing with correct dimensions
        inter_input = x.transpose(1, 2)  # [B, chunk_size, num_chunks, H]
        inter_input = inter_input.reshape(B*chunk_size, num_chunks, H)
        h_inter = h_inter[:, :B*chunk_size]  # Ensure correct batch size
        inter_output, h_inter = self.inter_gru(inter_input, h_inter)
        inter_output = inter_output.transpose(1, 2)
        inter_output = self.inter_norm(inter_output)
        inter_output = inter_output.transpose(1, 2)
        inter_output = inter_output.reshape(B, chunk_size, num_chunks, H)
        inter_output = inter_output.transpose(1, 2)
        
        x = x + inter_output
        return x, (h_intra, h_inter)

def causal_overlap_add(chunks, hop_size):
    """Causal overlap-add reconstruction."""
    B, num_sources, C, num_chunks, chunk_size = chunks.shape
    chunks = chunks.permute(0, 2, 1, 3, 4)
    
    # Only add past and present chunks (causal)
    T = (num_chunks - 1) * hop_size + chunk_size
    output = torch.zeros(B, C, num_sources, T, device=chunks.device)
    
    for i in range(num_chunks):
        start_idx = i * hop_size
        end_idx = start_idx + chunk_size
        output[:, :, :, start_idx:end_idx] += chunks[:, :, :, i, :]
        
    return output


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
