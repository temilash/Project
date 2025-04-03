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
        self.separator = DualPathGRUSeparator(
            input_channels=256,
            num_sources=self.num_sources,
            num_repeats=self.num_repeats,
            frame_length=self.frame_length,
            frame_step=self.frame_step            
        )
        self.decoder = SpectrogramDecoder(n_fft=n_fft, hop_length=hop_length)
    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        return length

    def forward(self, mixture):
        spectrogram, phase = self.encoder(mixture)
        masked_chunks = self.separator(spectrogram)
        input_length = mixture.size(-1)
        separated_waveforms = self.decoder(masked_chunks, phase, input_length)
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
        self.stft = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1)
        self.conv2d_freq = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, waveform):
        spectrogram = self.stft(waveform)
        phase = torch.angle(spectrogram)
        spectrogram = spectrogram.abs()
        spectrogram = self.conv2d_freq(spectrogram)
        return spectrogram, phase
        

class SpectrogramDecoder(nn.Module):
    def __init__(self, n_fft=512, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.reduce_conv = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, masked_spec, phase, input_length):
        B, C, S, freq, T_spec = masked_spec.shape
        T_phase = phase.size(-1)

        if T_spec < T_phase:
            diff = T_phase - T_spec
            masked_spec = F.pad(masked_spec, (0, diff))
        
        waveforms = []

        for s in range(S):
            spec_s = masked_spec[:, :, s, :, :]
            spec_s = self.reduce_conv(spec_s)
            complex_spec = torch.polar(spec_s.clamp(min=0), phase)
            b, ch, f, t = complex_spec.shape
            complex_spec = complex_spec.view(b * ch, f, t)
            waveform = torch.istft(complex_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=None, length=input_length)
            waveform = waveform.view(b, ch, -1)
            waveforms.append(waveform)
        
        return torch.stack(waveforms, dim=1)

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

class DualPathGRUSeparator(nn.Module):
    def __init__(self, input_channels=256, hidden_channels=128, num_sources=2,
                 num_repeats=4, frame_length=128, frame_step=64, inter_hidden=64):
        super().__init__()
        self.num_sources = num_sources
        self.num_repeats = num_repeats
        self.frame_length = frame_length
        self.frame_step = frame_step

        # Intra-chunk processing: same conv blocks + GRU layers as before
        self.conv_blocks = nn.ModuleList([
            CausalConv2D(
                in_channels=input_channels if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                kernel_size=(5, 3),
                dilation=(1, 1) if i % 2 == 0 else (1, 2)
            ) for i in range(num_repeats)
        ])
        self.cum_ln = CumulativeLayerNorm(hidden_channels)
        self.intra_gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_channels,
                hidden_size=hidden_channels,
                num_layers=2,
                batch_first=True,
                bidirectional=False
            ) for _ in range(num_repeats)
        ])
        self.mask_proj = nn.Conv2d(hidden_channels, input_channels * num_sources, kernel_size=1)

        # Inter-chunk processing: a causal GRU over chunk-level features.
        # We'll assume each chunk is summarized to a scalar per feature channel.
        # Here, input size = 1 (scalar per chunk), and we output a scaling factor.
        self.inter_gru = nn.GRU(input_size=1, hidden_size=inter_hidden, num_layers=1, batch_first=True, bidirectional=False)
        # Map the GRU output back to a scaling factor (using a linear layer)
        self.inter_linear = nn.Linear(inter_hidden, 1)

    def overlap_and_add(self, chunks, frame_step):
        N, B, C, S, F, K = chunks.shape
        outer_dims = (B, C, S, F)
        total_frames = frame_step * (N - 1) + K

        subframe_len = math.gcd(K, frame_step)
        subframes_per_chunk = K // subframe_len
        subframe_step = frame_step // subframe_len
        total_subframes = total_frames // subframe_len

        subframe_chunks = chunks.view(N, *outer_dims, subframes_per_chunk, subframe_len)
        subframe_chunks = subframe_chunks.permute(1, 2, 3, 4, 0, 5, 6)
        subframe_chunks = subframe_chunks.reshape(*outer_dims, -1, subframe_len)

        frame = torch.arange(0, total_subframes, device=chunks.device).unfold(0, subframes_per_chunk, subframe_step)
        frame = frame.contiguous().view(-1)[:subframe_chunks.shape[-2]]

        output = torch.zeros(*outer_dims, total_subframes, subframe_len, device=chunks.device)
        output.index_add_(-2, frame, subframe_chunks)
        full_spec = output.reshape(*outer_dims, -1)
        return full_spec

    def forward(self, spectrogram):
        B, C, freq, T_in = spectrogram.shape
        # --- Split the spectrogram into overlapping chunks along time ---
        # Shape after unfold: [B, C, freq, n_chunks, frame_length]
        chunks = spectrogram.unfold(-1, self.frame_length, self.frame_step)
        n_chunks = chunks.size(-2)
        # Permute to bring the chunk dimension first: [n_chunks, B, C, freq, frame_length]
        chunks = chunks.permute(3, 0, 1, 2, 4)

        intra_outputs = []
        # Process each chunk independently (intra-chunk)
        for chunk in chunks:
            out = chunk  # shape: [B, C, freq, frame_length]
            for conv in self.conv_blocks:
                out = conv(out)
            out = self.cum_ln(out)
            H = out.size(1)  # hidden channels
            # Rearrange for GRU processing along time within the chunk:
            # Permute to [B, freq, H, frame_length] then reshape to [B*freq, frame_length, H]
            out = out.permute(0, 2, 1, 3).contiguous()
            T_chunk = out.size(-1)
            out = out.view(B * freq, T_chunk, H)
            for gru in self.intra_gru_layers:
                out, _ = gru(out)
            T_new = out.size(1)
            # Reshape back: [B, freq, T_new, H] then permute to [B, H, freq, T_new]
            out = out.view(B, freq, T_new, H).permute(0, 3, 1, 2).contiguous()
            # Project to mask space
            mask = self.mask_proj(out)
            mask = mask.view(B, self.num_sources, C, freq, T_new)
            mask = F.relu(mask)
            # Adjust chunk if necessary
            if self.frame_length != T_new:
                chunk = chunk[..., :T_new]
            masked_chunk = chunk.unsqueeze(1) * mask  # [B, num_sources, C, freq, T_new]
            # Permute to [B, C, num_sources, freq, T_new]
            masked_chunk = masked_chunk.permute(0, 2, 1, 3, 4)
            intra_outputs.append(masked_chunk)
        # Now intra_outputs is a list of length n_chunks, each with shape [B, C, num_sources, freq, T_new]
        # Stack along a new dimension (chunk dimension):
        intra_outputs = torch.stack(intra_outputs, dim=0)  # [n_chunks, B, C, num_sources, freq, T_new]

        # --- Inter-chunk processing ---
        # For each chunk, compute a summary statistic (e.g., average over time) to obtain a chunk-level feature.
        chunk_summary = intra_outputs.mean(dim=-1)  # [n_chunks, B, C, num_sources, freq]
        # For simplicity, collapse all dimensions except the chunk dimension and treat each chunk's summary as a scalar.
        # Here, we average over (C, num_sources, freq):
        chunk_summary = chunk_summary.mean(dim=(2, 3, 4))  # [n_chunks, B]
        # Transpose so that we have [B, n_chunks, 1] as the input to inter-chunk GRU:
        chunk_summary = chunk_summary.transpose(0, 1).unsqueeze(-1)  # [B, n_chunks, 1]

        # Process with a causal (unidirectional) inter-chunk GRU
        inter_out, _ = self.inter_gru(chunk_summary)  # [B, n_chunks, inter_hidden]
        # Map GRU output to a scaling factor:
        scaling = self.inter_linear(inter_out)  # [B, n_chunks, 1]
        # Reshape scaling to apply per chunk: expand to match intra_outputs shape.
        scaling = scaling.transpose(0, 1)  # [n_chunks, B, 1]
        # Expand to shape [n_chunks, B, C, num_sources, freq, 1]
        scaling = scaling.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(n_chunks, B, C, self.num_sources, freq, 1)

        # Apply the inter-chunk scaling factor to the intra-chunk outputs
        inter_processed = intra_outputs * scaling  # [n_chunks, B, C, num_sources, freq, T_new]

        # Finally, reconstruct the full spectrogram via overlap-and-add.
        full_masked_spec = self.overlap_and_add(inter_processed, self.frame_step)
        return full_masked_spec


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
