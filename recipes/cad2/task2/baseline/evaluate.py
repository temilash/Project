"""Evaluate the enhanced signals using the HAAQI metric."""

from __future__ import annotations

# pylint: disable=import-error
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pyloudnorm as pyln
from numpy import ndarray
from omegaconf import DictConfig
import scipy.signal as signal

from clarity.enhancer.multiband_compressor import MultibandCompressor
from clarity.evaluator.haaqi import compute_haaqi
from clarity.utils.audiogram import Listener
from clarity.utils.flac_encoder import read_flac_signal
from clarity.utils.results_support import ResultsFile
from clarity.utils.signal_processing import compute_rms, resample

logger = logging.getLogger(__name__)


def apply_frequency_gains(signal_data, sample_rate, frequencies, gains):
    """Apply frequency-specific gains using a parametric equalizer while avoiding over-amplification.

    Args:
        signal_data (ndarray): Input audio signal (1D NumPy array).
        sample_rate (int): Sample rate of the audio.
        frequencies (list): Frequency bands in Hz.
        gains (list): Gain values in dB for each frequency.

    Returns:
        ndarray: Audio signal with frequency-based gain adjustments.
    """
    output_signal = np.zeros_like(signal_data)  # Initialize empty signal

    for freq, gain in zip(frequencies, gains):
        if gain == 0:  # Skip frequencies with no gain adjustment
            continue

        # Create a peak filter (band-pass filter centered at `freq`)
        b, a = signal.iirpeak(freq / (sample_rate / 2), Q=1.0)
        filtered_signal = signal.lfilter(b, a, signal_data)

        # ? Accumulate the frequency-shaped signal instead of direct gain boosting
        output_signal += filtered_signal  

    return output_signal


def apply_gains(stems: dict, sample_rate: float, gains: dict, listener: dict) -> dict:
    """Apply instrument and listener-specific gains before remixing.

    Args:
        stems (dict): Dictionary of instrument stems (stereo signals).
        sample_rate (float): Sample rate of the signal.
        gains (dict): Dictionary containing instrument gain values.
        listener (Listener): Listener object with audiogram information.

    Returns:
        dict: Dictionary of stems with correctly applied gains.
    """
    meter = pyln.Meter(int(sample_rate))

    # Extract listener frequency bands and gain values
    frequencies = listener.audiogram_left.frequencies
    gain_left_freq = listener.audiogram_left.levels
    gain_right_freq = listener.audiogram_right.levels

    stems_gain = {}  # Initialize dictionary to store updated stems

    for stem_str, stem_signal in stems.items():
        if stem_signal.shape[0] < stem_signal.shape[1]:
            stem_signal = stem_signal.T

        # Compute current loudness for left and right channels
        left_lufs = meter.integrated_loudness(stem_signal[:, 0])
        right_lufs = meter.integrated_loudness(stem_signal[:, 1])

        # Handle silent signals
        if left_lufs == -np.inf:
            left_lufs = -80
        if right_lufs == -np.inf:
            right_lufs = -80

        # Retrieve instrument gain (default to 0 dB if missing)
        instrument_gain = gains.get(stem_str, 0)

        # Apply instrument gain
        left_gain = left_lufs + instrument_gain
        right_gain = right_lufs + instrument_gain

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Possible clipped samples in output")
            adjusted_left = pyln.normalize.loudness(stem_signal[:, 0], left_lufs, left_gain)
            adjusted_right = pyln.normalize.loudness(stem_signal[:, 1], right_lufs, right_gain)

        # Apply frequency-dependent listener-specific gains
        adjusted_left = apply_frequency_gains(adjusted_left, sample_rate, frequencies, gain_left_freq)
        adjusted_right = apply_frequency_gains(adjusted_right, sample_rate, frequencies, gain_right_freq)

        stems_gain[stem_str] = np.stack([adjusted_left, adjusted_right], axis=1)

    #  Ensure all required stems are present before returning
    if not stems_gain:
        raise ValueError("Error: No stems were processed in apply_gains()!")

    return stems_gain



def remix_stems(stems: dict) -> ndarray:
    """Remix the stems into a stereo signal.

    The remixing is done by summing the stems.

    Args:
        stems (dict): Dictionary of stems.

    Returns:
        ndarray: Stereo signal.
    """
    remix_signal = np.zeros(stems["source_1"].shape)
    for _, stem_signal in stems.items():
        remix_signal += stem_signal
    return remix_signal

def make_scene_listener_list(scenes_listeners: dict, small_test: bool = False) -> list:
    """Make the list of scene-listener pairing to process

    Args:
        scenes_listeners (dict): Dictionary of scenes and listeners.
        small_test (bool): Whether to use a small test set.

    Returns:
        list: List of scene-listener pairings.

    """
    scene_listener_pairs = [
        (scene, listener)
        for scene in scenes_listeners
        for listener in scenes_listeners[scene]
    ]

    # Can define a standard 'small_test' with just 1/50 of the data
    if small_test:
        scene_listener_pairs = scene_listener_pairs[::400]

    return scene_listener_pairs


def set_song_seed(song: str) -> None:
    """Set a seed that is unique for the given song"""
    song_encoded = hashlib.md5(song.encode("utf-8")).hexdigest()
    song_md5 = int(song_encoded, 16) % (10**8)
    np.random.seed(song_md5)



def load_reference_stems(music_dir: str | Path, stems: dict) -> dict[Any, ndarray]:
    """Load the reference stems for a given scene.

    Args:
        music_dir (str | Path): Path to the music directory.
        stems (dict): Dictionary of stems
    Returns:
        dict: Dictionary of reference stems.
    """
    reference_stems= {}
    for source_id, source_data in stems.items():
        if source_id == "mixture":
            continue

        stem, _= read_flac_signal(Path(music_dir) / source_data["track"])
        reference_stems[source_id]= stem

    return reference_stems


def adjust_level(signal: np.ndarray, gains_scene: dict) -> np.ndarray:
    """
    Adjust the level of the signal to compensate the effect of amplifying the
    sources
    """
    dbi= np.array(list(gains_scene.values()))
    dbn= -10 * np.log10(np.sum(10 ** (dbi / 10)) / dbi.shape[0])
    return signal * 10 ** (dbn / 20)


@ hydra.main(config_path="", config_name="config", version_base=None)
def run_calculate_aq(config: DictConfig) -> None:
    """Evaluate the enhanced signals using the HAAQI metric."""

    enhanced_folder= Path("enhanced_signals")
    logger.info(f"Evaluating from {enhanced_folder} directory")

    # Load listener audiograms and songs
    listener_dict= Listener.load_listener_dict(config.path.listeners_file)

    with Path(config.path.gains_file).open("r", encoding="utf-8") as file:
        gains= json.load(file)

    with Path(config.path.scenes_file).open("r", encoding="utf-8") as file:
        scenes= json.load(file)

    with Path(config.path.scene_listeners_file).open("r", encoding="utf-8") as file:
        scenes_listeners= json.load(file)

    with Path(config.path.music_file).open("r", encoding="utf-8") as file:
        songs= json.load(file)

        # Load compressor params
    with Path(config.path.enhancer_params_file).open("r", encoding="utf-8") as file:
        enhancer_params= json.load(file)

    enhancer= MultibandCompressor(
        crossover_frequencies = config.enhancer.crossover_frequencies,
        sample_rate = config.input_sample_rate,
    )

    scores_headers= [
        "scene",
        "song",
        "listener",
        "left_haaqi",
        "right_haaqi",
        "avg_haaqi",
    ]
    if config.evaluate.batch_size == 1:
        results_file= ResultsFile(
            "scores.csv",
            header_columns = scores_headers,
        )
    else:
        results_file = ResultsFile(
            f"scores_{config.evaluate.batch + 1}-{config.evaluate.batch_size}.csv",
            header_columns=scores_headers,
        )

    scene_listener_pairs = make_scene_listener_list(
        scenes_listeners, config.evaluate.small_test
    )
    scene_listener_pairs = scene_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]
    num_scenes = len(scene_listener_pairs)
    for idx, scene_listener_pair in enumerate(scene_listener_pairs, 1):
        scene_id, listener_id = scene_listener_pair

        scene = scenes[scene_id]
        song_name = scene["music"]

        logger.info(
            f"[{idx:03d}/{num_scenes:03d}] "
            f"Evaluating {scene_id} for listener {listener_id}"
        )
        
        # Evaluate listener
        listener = listener_dict[listener_id]
        
        # Load reference signals
        reference_stems = load_reference_stems(
            Path(config.path.music_dir), songs[song_name]
        )
        reference_stems = apply_gains(reference_stems, config.input_sample_rate, gains[scene["gain"]], listener)

        reference_mixture = remix_stems(reference_stems)
        reference_mixture = adjust_level(reference_mixture, gains[scene["gain"]])

        # Set the random seed for the scene
        if config.evaluate.set_random_seed:
            set_song_seed(scene_id)

        

        # Compressor params
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

        # Get set directory
        if 0 < int(scene_id[1:]) < 49999:
            dataset_dir = "train"
        elif 50000 < int(scene_id[1:]) < 59999:
            dataset_dir = "valid"
        else:
            dataset_dir = "test"

        # Load enhanced signal
        enhanced_signal, _ = read_flac_signal(
            Path(enhanced_folder) / dataset_dir / f"{scene_id}_{listener.id}_remix.flac"
        )

        # Apply hearing aid to reference signals
        enhancer.set_compressors(**mbc_params_listener["left"])
        left_reference = enhancer(
            signal=reference_mixture[:, 0],
        )
        enhancer.set_compressors(**mbc_params_listener["right"])
        right_reference = enhancer(signal=reference_mixture[:, 1])

        # Compute the scores
        left_score = compute_haaqi(
            processed_signal=resample(
                enhanced_signal[:, 0],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                left_reference[0], config.input_sample_rate, config.HAAQI_sample_rate
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=listener.audiogram_left,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 0])),
        )

        right_score = compute_haaqi(
            processed_signal=resample(
                enhanced_signal[:, 1],
                config.remix_sample_rate,
                config.HAAQI_sample_rate,
            ),
            reference_signal=resample(
                right_reference[0], config.input_sample_rate, config.HAAQI_sample_rate
            ),
            processed_sample_rate=config.HAAQI_sample_rate,
            reference_sample_rate=config.HAAQI_sample_rate,
            audiogram=listener.audiogram_right,
            equalisation=2,
            level1=65 - 20 * np.log10(compute_rms(reference_mixture[:, 1])),
        )

        # Save scores
        results_file.add_result(
            {
                "scene": scene_id,
                "song": song_name,
                "listener": listener.id,
                "left_haaqi": left_score,
                "right_haaqi": right_score,
                "avg_haaqi": float(np.mean([left_score, right_score])),
            }
        )

    logger.info("Done!")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    run_calculate_aq()
