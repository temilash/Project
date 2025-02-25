""" Run the dummy enhancement. """

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

# pylint: disable=import-error
import hydra
import numpy as np
import torch
from numpy import ndarray
from omegaconf import DictConfig
from torchaudio.transforms import Fade

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.utils.audiogram import Listener
from clarity.utils.flac_encoder import read_flac_signal, save_flac_signal
from clarity.utils.source_separation_support import get_device
from recipes.cad2.task2.baseline.evaluate import (
    adjust_level,
    apply_gains,
    make_scene_listener_list,
    remix_stems,
)
from recipes.cad2.task2.ConvTasNet.local.tasnet import ConvTasNetStereo

logger = logging.getLogger(__name__)


def check_repeated_source(gains: dict, source_list: dict) -> dict:
    """Check if mixture has 2 voices of the same instrument.
    Apply average gain to both voices.

    Args:
        gains (dict): Dictionary of original gains.
        source_list (dict): Dictionary of sources in mixture.

    Returns:
        dict: Dictionary of modified gains.
    """
    count_dict = Counter(source_list.values())
    two_voices = [key for key, value in source_list.items() if count_dict[value] > 1]
    two_voices_gain = [gain for source, gain in gains.items() if source in two_voices]
    two_voices_gain = np.mean(two_voices_gain)

    new_gains = {}
    for key, value in gains.items():
        if key in two_voices:
            new_gains[key] = two_voices_gain
        else:
            new_gains[key] = value
    return new_gains


def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input audio to have a consistent loudness before separation.
    
    Args:
        audio (torch.Tensor): Input mixture signal (batch, channels, time)

    Returns:
        torch.Tensor: Normalized audio
    """
    mean_val = torch.mean(audio, dim=-1, keepdim=True)
    std_val = torch.std(audio, dim=-1, keepdim=True) + 1e-8  # Avoid division by zero
    return (audio - mean_val) / std_val

def apply_phase_correction(reference_signal, target_signal):
    """
    Aligns the phase of the target signal to match the reference signal.

    Args:
        reference_signal (np.ndarray): The reference signal.
        target_signal (np.ndarray): The signal to be phase-aligned.

    Returns:
        np.ndarray: Phase-aligned target signal.
    """
    analytic_ref = hilbert(reference_signal)
    analytic_target = hilbert(target_signal)

    correlation = np.correlate(np.abs(analytic_ref), np.abs(analytic_target), mode="full")
    delay = np.argmax(correlation) - (len(reference_signal) - 1)

    if delay > 0:
        aligned_signal = np.pad(target_signal, (delay, 0), mode="constant")[:-delay]
    elif delay < 0:
        aligned_signal = np.pad(target_signal, (0, -delay), mode="constant")[-delay:]
    else:
        aligned_signal = target_signal

    return aligned_signal

def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.5,  # Increased overlap for smoother blending
    device: torch.device | str | None = None,
):
    """
    Apply model to a given mixture using improved Overlap-Add processing with Hann window.

    Args:
        model (torch.nn.Module): Model to use for separation.
        mix (torch.Tensor): Mixture to separate, shape (batch, channels, time).
        sample_rate (int): Sampling rate of the mixture.
        segment (float): Segment length in seconds.
        overlap (float): Overlap percentage (0 to 1).
        device (torch.device, str, or None): Processing device.

    Returns:
        torch.Tensor: Estimated sources with improved overlap-add.
    """
    device = mix.device if device is None else torch.device(device)
    mix = mix.to(device)

    if mix.ndim == 2:
        mix = mix.unsqueeze(0)  # Convert stereo signal to batch format

    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment)
    step_size = int(chunk_len * (1 - overlap))  # Step size based on overlap
    fade_window = torch.tensor(hann(chunk_len, sym=False), device=device)

    # Normalize input before separation
    mix = normalize_audio(mix)

    final = torch.zeros(batch, 4, channels, length, device=device)
    sum_weights = torch.zeros_like(final)

    start = 0
    while start < length - step_size:
        end = min(start + chunk_len, length)
        chunk = mix[:, :, start:end]

        # Apply model for source separation
        with torch.no_grad():
            out = model.forward(chunk)

        # Apply Hann window for smooth blending
        out *= fade_window[: out.shape[-1]]

        final[:, :, :, start:end] += out
        sum_weights[:, :, :, start:end] += fade_window[: out.shape[-1]]

        start += step_size  # Move by step size

    # Normalize to avoid artifacts
    final /= torch.clamp(sum_weights, min=1e-8)

    # Convert to NumPy and apply phase correction
    final_np = final.cpu().detach().numpy()

    for i in range(final_np.shape[1]):  # Loop over separated sources
        for j in range(final_np.shape[2]):  # Left and Right channels
            final_np[:, i, j, :] = apply_phase_correction(mix.cpu().numpy()[0, j, :], final_np[:, i, j, :])

    return final_np


# pylint: disable=unused-argument
def decompose_signal(
    model: dict[str, ConvTasNetStereo],
    signal: ndarray,
    signal_sample_rate: int,
    device: torch.device,
    sources_list: dict,
    listener: Listener,
    add_residual: float = 0.0,
) -> dict[str, ndarray]:
    """
    Decompose the signal into the estimated sources.

    The listener is ignored by the baseline system as it
     is not performing personalised decomposition.

    Args:
        model (dict): Dictionary of separation models.
        model_sample_rate (int): Sample rate of the separation model.
        signal (ndarray): Signal to decompose.
        signal_sample_rate (int): Sample rate of the signal.
        device (torch.device): Device to use for separation.
        sources_list (dict): List of sources to separate.
        listener (Listener): Listener audiogram.
        add_residual (float): Add residual to the target estimated sources from
            the accompaniment.

    Returns:
        dict: Dictionary of estimated sources.
    """
    out_sources = {}
    for source in sources_list:
        est_sources = separate_sources(
            model=model[sources_list[source]],
            mix=signal,
            sample_rate=signal_sample_rate,
            number_sources=2,
            device=device,
        )
        target, accompaniment = est_sources.squeeze(0).cpu().detach().numpy()

        if add_residual > 0.0:
            target += accompaniment * add_residual

        out_sources[source] = target.T
    return out_sources


def load_separation_model(
    causality: str, device: torch.device, force_redownload: bool = True
) -> dict[str, ConvTasNetStereo]:
    """
    Load the separation model.
    Args:
        causality (str): Causality of the model (causal or noncausal).
        device (torch.device): Device to load the model.
        force_redownload (bool): Whether to force redownload the model.

    Returns:
        model: Separation model.
    """
    models = {}
    causal = {"causal": "Causal", "noncausal": "NonCausal"}

    for instrument in [
        "Bassoon",
        "Cello",
        "Clarinet",
        "Flute",
        "Oboe",
        "Sax",
        "Viola",
        "Violin",
    ]:
        logger.info(
            "Loading model "
            f"cadenzachallenge/ConvTasNet_{instrument}_{causal[causality]}"
        )
        models[instrument] = ConvTasNetStereo.from_pretrained(
            f"cadenzachallenge/ConvTasNet_{instrument}_{causal[causality]}",
            force_download=force_redownload,
        ).to(device)
    return models


def process_remix_for_listener(
    signal: ndarray, enhancer: MultibandCompressor, enhancer_params: dict, listener
) -> ndarray:
    """Process the stems from sources.

    Args:

    Returns:
        ndarray: Processed signal.
    """

    output = []
    for side, ear in enumerate(["left", "right"]):
        enhancer.set_compressors(**enhancer_params[ear])
        output.append(enhancer(signal[:, side]))

    return np.vstack(output).T


@hydra.main(config_path="", config_name="config", version_base=None)
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into the estimated sources.
    Next, applies the target gain per source.
    Then, remixes the sources to create the enhanced signal.
    Finally, the enhanced signal is amplified for the listener.

    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    """
    if config.separator.causality not in ["causal", "noncausal"]:
        raise ValueError(
            f"Causality must be causal or noncausal, {config.separator.causality} was"
            " provided."
        )

    # Set the output directory where processed signals will be saved
    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    device, _ = get_device(config.separator.device)

    # Load listener audiograms and songs
    listener_dict = Listener.load_listener_dict(config.path.listeners_file)

    # Load gains
    with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
        gains = json.load(file)

    # Load Scenes
    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    # Load scene listeners
    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners = json.load(file)

    # load songs
    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs = json.load(file)

    # Load compressor params
    with Path(config.path.enhancer_params_file).open("r", encoding="utf-8") as file:
        enhancer_params = json.load(file)

    # Load separation model
    separation_models = load_separation_model(
        config.separator.causality, device, config.separator.force_redownload
    )

    # create hearing aid
    enhancer = MultibandCompressor(
        crossover_frequencies=config.enhancer.crossover_frequencies,
        sample_rate=config.input_sample_rate,
    )

    # Select a batch to process
    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )

    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch:: config.evaluate.batch_size
    ]

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    num_scenes = len(scene_listener_pairs)
    for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = scene["music"]

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Processing {scene_id}: song {song_name} for listener {listener_id}"
        )
        # Get the listener's audiogram
        listener = listener_dict[listener_id]

        # Get the listener's compressor params
        mbc_params_listener: dict[str, dict] = {"left": {}, "right": {}}

        for ear in ["left", "right"]:
            mbc_params_listener[ear]["release"] = config.enhancer.release
            mbc_params_listener[ear]["attack"] = config.enhancer.attack
            mbc_params_listener[ear]["threshold"] = config.enhancer.threshold
        mbc_params_listener["left"]["ratio"] = enhancer_params[listener_id]["cr_l"]
        mbc_params_listener["right"]["ratio"] = enhancer_params[listener_id]["cr_r"]
        mbc_params_listener["left"]["makeup_gain"] = enhancer_params[listener_id][
            "gain_l"
        ]
        mbc_params_listener["right"]["makeup_gain"] = enhancer_params[listener_id][
            "gain_r"
        ]

        # Read the mixture signal
        # Convert to 32-bit floating point and transpose
        # from [samples, channels] to [channels, samples]
        source_list = {
            f"source_{idx}": s["instrument"].split("_")[0]
            for idx, s in enumerate(songs[song_name].values(), 1)
            if "Mixture" not in s["instrument"]
        }

        mixture_signal, mix_sample_rate = read_flac_signal(
            filename=Path(config.path.music_dir) / songs[song_name]["mixture"]["track"]
        )
        assert mix_sample_rate == config.input_sample_rate

        start = songs[song_name]["mixture"]["start"]
        end = start + songs[song_name]["mixture"]["duration"]
        mixture_signal = mixture_signal[
            int(start * mix_sample_rate): int(end * mix_sample_rate),
            :,
        ]

        # Estimate the isolated sources
        stems: dict[str, ndarray] = decompose_signal(
            model=separation_models,
            signal=mixture_signal.T,  # Channel first
            signal_sample_rate=config.input_sample_rate,
            device=device,
            sources_list=source_list,
            listener=listener,
            add_residual=config.separator.add_residual,
        )

        # Apply gains to sources
        gain_scene = check_repeated_source(gains[scene["gain"]], source_list)
        stems = apply_gains(stems, config.input_sample_rate, gain_scene, listener)

        # Downmix to stereo
        enhanced_signal = remix_stems(stems)

        # adjust levels to get roughly -40 dB before compressor
        enhanced_signal = adjust_level(enhanced_signal, gains[scene["gain"]])

        # Apply compressor
        enhanced_signal = process_remix_for_listener(
            signal=enhanced_signal,
            enhancer=enhancer,
            enhancer_params=mbc_params_listener,
            listener=listener,
        )

        # Save the enhanced signal in the corresponding directory
        if 0 < int(scene_id[1:]) < 49999:
            out_dir = "train"
        elif 50000 < int(scene_id[1:]) < 59999:
            out_dir = "valid"
        else:
            out_dir = "test"

        filename = (
            Path(enhanced_folder) / out_dir / f"{scene_id}_{listener.id}_remix.flac"
        )

        filename.parent.mkdir(parents=True, exist_ok=True)
        save_flac_signal(
            signal=enhanced_signal,
            filename=filename,
            signal_sample_rate=config.input_sample_rate,
            output_sample_rate=config.remix_sample_rate,
            do_clip_signal=True,
            do_soft_clip=config.soft_clip,
        )

    logger.info("Done!")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()
