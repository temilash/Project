# Exploring Machine Learning for Hearing Impairment

## Overview
This project explores how **machine learning can enhance classical music for listeners with hearing impairments**.  
It builds on the [Cadenza Challenge](https://cadenzachallenge.org/) by developing a **spectrogram-based Dual-Path RNN (DPRNN) separator** to improve instrument separation in classical music recordings.  

Unlike the Conv-TasNet baseline, the model preserves both magnitude and phase using complex-ratio masks, enabling clearer remixes of instrument mixtures. The system also applies **listener-specific frequency gains** (based on audiograms) and enforces a **5 ms causal lookahead** to meet real-time processing constraints.

## Key Features
- 🎵 **Spectrogram-based DPRNN** for instrument separation  
- 🎚️ **Listener-specific gain application** using audiograms  
- ⏱️ **Causal pipeline** with 5 ms lookahead for real-time suitability  
- 📊 Evaluated using **HAAQI** (Hearing-Aid Audio Quality Index) and separation metrics  

## Results
- Achieved a **modest HAAQI improvement** over the Conv-TasNet baseline  
- Demonstrated clearer remixes and better preservation of instrument timbre  
- Identified limitations in dataset diversity, causal artefacts, and phase modelling  

## Future Work
- Explore **transformer-based architectures** for better long-range modelling  
- Improve **phase reconstruction** beyond complex ratio masks  
- Expand datasets with **live ensemble recordings**  
- Run **listening tests** to validate perceptual benefits  
- Investigate deployment on **resource-constrained hearing aid devices**

---



⚡ *This project demonstrates how tailored machine learning pipelines can make classical music more accessible to hearing-impaired listeners, laying the groundwork for future hearing-aid innovations.*

Look at recipes/cad2/task2 for my code
