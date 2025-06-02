# MS Music: Mass Spectrometry Data Sonification

**Version: 0.1.0**

`ms_music` is a Python package designed to transform mass spectrometry data from `.mzML` files into audible sound. This process, known as sonification, can provide a novel way to explore and interpret complex scientific data. The package offers various methods for sound generation and allows for the application of several audio effects to customize the output.

## Table of Contents

1.  [Features](#features)
2.  [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Steps](#steps)
3.  [Quick Start](#quick-start)
4.  [Detailed Usage](#detailed-usage)
    * [Initialization](#initialization)
    * [Loading and Preprocessing Data](#loading-and-preprocessing-data)
    * [Sonification Methods](#sonification-methods)
    * [Applying Audio Effects via `sonifier.apply_effect`](#applying-audio-effects-via-sonifierapply_effect)
        * [Standard Effects](#standard-effects)
        * [Experimental Effects](#experimental-effects)
    * [Using Additional Creative Effects Directly](#using-additional-creative-effects-directly)
    * [Saving Audio](#saving-audio)
5.  [Examples in Jupyter Notebook](#examples-in-jupyter-notebook)
6.  [Contributing](#contributing)
7.  [License](#license)

## Features

* **MZML File Input**: Supports loading of standard `.mzML` files.
* **MS Level Selection**: Process data from MS1 or MS2 levels.
* **Data Preprocessing**: Rounds m/z values, groups intensities, and normalizes data.
* **Flexible Sonification Strategies**:
    * **Gradient Method**: Continuous sine waves for each m/z, amplitude modulated by intensity with smooth transitions.
    * **ADSR Method**: Each scan is an event shaped by an ADSR envelope, suitable for rhythmic or articulated sounds.
* **Audio Effects Suite**:
    * **Standard Effects (via `sonifier.apply_effect`)**: HPSS, Notch Filter, Butterworth Filter, Chebyshev Type I Filter.
    * **Experimental Effects (via `sonifier.apply_effect`)**: Functional implementations derived from original scripts, including a time-varying PLL filter, LMS modulation, spectral analysis FM, phase vocoder (pitch/time stretch), and FFT-based filtering. Use with awareness of their experimental nature.
    * **Additional Creative Effects (direct use from `additional_sound_modifiers.py`)**: Granular synthesis, pitch shifting, time stretching, algorithmic reverb, and chorus effects.
* **Customizable Output**: Control total audio duration and sample rate.
* **WAV File Export**: Save sonified audio as `.wav` files.

## Installation

### Prerequisites

* Python 3.8 or higher.
* `pip` for package installation.

### Steps

1.  **Clone the repository or download the package files:**
    ```bash
    # If you have a git repository for ms_music:
    # git clone <your_repository_url_for_ms_music>
    # cd ms_music
    # Otherwise, ensure all package files (setup.py, ms_music/) are in a directory.
    ```
    Navigate to the root directory of the package (where `setup.py` is located).

2.  **Install the package and its dependencies:**
    From the root directory, run:
    ```bash
    pip install .
    ```
    This command installs `ms_music` and automatically handles dependencies: `numpy`, `pandas`, `matchms>=0.25.0`, `wavio`, `librosa>=0.9.0`, `scipy`, `tqdm`.

## Quick Start

```python
from ms_music import MSSonifier
import os

# Configuration
mzml_file = "path/to/your/data.mzML" # Replace with your actual file path
output_dir = "audio_output_quickstart"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(mzml_file):
    print(f"Error: mzML file not found at {mzml_file}. Please provide a valid path.")
else:
    # Initialize
    sonifier = MSSonifier(
        filepath=mzml_file,
        ms_level=1,
        total_duration_minutes=0.25, # Keep duration short for quick tests (15 seconds)
        sample_rate=44100
    )

    # Load and preprocess
    sonifier.load_and_preprocess_data()

    if sonifier.processed_spectra_dfs: # Check if data processing was successful
        # Sonify using the gradient method
        sonifier.sonify(method='gradient')

        # Apply a simple low-pass filter
        sonifier.apply_effect('butterworth_filter', 
                              effect_params={'cutoff_freq': 1200, 'btype': 'low'})

        # Save the audio
        output_path = os.path.join(output_dir, "quick_start_ms_sound.wav")
        sonifier.save_audio(output_path)
        print(f"Sonification complete! Audio saved to {output_path}")
    else:
        print("Data loading or preprocessing failed. Cannot sonify.")
```

## Detailed Usage

### Initialization

```python
from ms_music import MSSonifier

sonifier = MSSonifier(
    filepath="path/to/your/data.mzML",
    ms_level=1,  # MS level (1 or 2)
    total_duration_minutes=1.0,  # Desired audio duration in minutes
    sample_rate=44100  # Standard audio sample rate
)
```

### Loading and Preprocessing Data

Load the `.mzML` file and preprocess the spectral data.
```python
# Process all scans from the specified MS level
sonifier.load_and_preprocess_data()

# Or, process a specific segment (e.g., scans from 25% to 75% of the file)
# sonifier.load_and_preprocess_data(scan_ratio_range=(0.25, 0.75))
```

### Sonification Methods

Generate base audio using one of the available methods:
* **`gradient` method**: `sonifier.sonify(method='gradient', method_params={'overlap_percentage': 0.05})`
    * `overlap_percentage` (float, 0.0 to 1.0): Controls intensity fade duration between scans.
* **`adsr` method**: `sonifier.sonify(method='adsr', method_params={'adsr_settings': adsr_config_dict})`
    * `adsr_config_dict` (dict): Specifies `'attack_time_pc'`, `'decay_time_pc'`, `'sustain_level_pc'`, `'release_time_pc'`, and `'randomize'` (bool). See `audio_generator.py` or notebook for defaults.

### Applying Audio Effects via `sonifier.apply_effect`

Effects modify the `sonifier.current_audio_data`. They are applied sequentially if multiple calls are made.

#### Standard Effects

* **HPSS (Harmonic-Percussive Source Separation)**:
  `sonifier.apply_effect('hpss', effect_params={'margin': 16, 'harmonic': True, 'percussive': False})`
    * `margin` (float): Separation margin.
    * `harmonic` (bool): Output harmonic component.
    * `percussive` (bool): Output percussive component.
* **Notch Filter**:
  `sonifier.apply_effect('notch_filter', effect_params={'notch_freq': 1000, 'quality_factor': 10})`
    * `notch_freq` (float): Center frequency to attenuate (Hz).
    * `quality_factor` (float): Q-factor (bandwidth of notch).
* **Butterworth Filter**:
  `sonifier.apply_effect('butterworth_filter', effect_params={'cutoff_freq': 800, 'btype': 'low', 'order': 4})`
    * `cutoff_freq` (float or tuple): Cutoff(s) in Hz.
    * `btype` (str): `'low'`, `'high'`, `'bandpass'`, `'bandstop'`.
    * `order` (int): Filter order.
    * `gustafson_method` (bool): Use Gustafson's method for `filtfilt`.
* **Chebyshev Type I Filter**:
  `sonifier.apply_effect('chebyshev1_filter', effect_params={'cutoff_freq': 1500, 'ripple_db': 1, 'btype': 'low', 'order': 4})`
    * `cutoff_freq` (float or tuple): Cutoff(s) in Hz.
    * `ripple_db` (float): Max passband ripple in dB (must be > 0).
    * `btype` (str): Filter type.
    * `order` (int): Filter order.

#### Experimental Effects
These effects are functional implementations based on the original `complex.py` script. They may produce unique sounds but can be sensitive to parameters and input audio. **Use with exploration in mind.**

* **PLL Time-Varying Filter**: `sonifier.apply_effect('experimental_pll_filter', effect_params={'center_freq': 500, 'loop_bandwidth_hz': 20, ...})`
    * `center_freq` (float): Initial NCO frequency.
    * `loop_bandwidth_hz` (float): PLL loop responsiveness.
    * `damping_factor` (float): PLL loop damping.
    * `output_filter_freq_offset_hz` (float): Offset for the output LPF cutoff from NCO frequency.
    * `output_filter_min_freq_hz`, `output_filter_max_freq_hz` (float): Clamping for output LPF cutoff.
* **LMS Modulation**: `sonifier.apply_effect('experimental_lms_modulation', effect_params={'n_segments': 100, 'mu': 0.05})`
    * `n_segments` (int): Number of audio segments for LMS adaptation.
    * `mu` (float): LMS adaptation rate.
* **Spectral Analysis FM**: `sonifier.apply_effect('experimental_spectral_analysis_fm', effect_params={})`
    * Uses Welch's method to find dominant frequency for FM modulation. Parameters are mostly internal.
* **Phase Vocoder (Pitch/Time)**: `sonifier.apply_effect('experimental_phase_vocoder_modulation', effect_params={'stretch_factor': 1.0, 'pitch_shift_semitones': 0.0})`
    * `n_fft` (int): FFT window size.
    * `hop_length` (int): Hop length for STFT.
    * `stretch_factor` (float): Time stretch factor (>1 slows, <1 speeds up).
    * `pitch_shift_semitones` (float): Pitch shift in semitones.
* **FFT Filter**: `sonifier.apply_effect('experimental_fft_filter', effect_params={'threshold_factor': 0.01})`
    * `threshold_factor` (float): Relative threshold (0-1) for zeroing out FFT coefficients based on max coefficient.

### Using Additional Creative Effects Directly
These effects are provided in `ms_music/additional_sound_modifiers.py`. They are used by importing the module and calling the functions directly on a NumPy audio array (e.g., obtained from `sonifier.get_current_audio()`).

```python
from ms_music import additional_sound_modifiers as creative_fx
# ... (after sonifier has generated some audio)
current_audio = sonifier.get_current_audio() 
if current_audio is not None:
    # Example: Apply granular synthesis
    granular_audio = creative_fx.apply_granular_synthesis(
        current_audio, 
        sonifier.sample_rate, 
        grain_duration_ms=40, 
        density=1.5,
        pitch_variation_semitones=1.0
    )
    # 'granular_audio' can now be saved or further processed
    # sonifier.current_audio_data = granular_audio # Optionally update sonifier's buffer
    # sonifier.save_audio("granular_output.wav")
```
* **Granular Synthesis**: `creative_fx.apply_granular_synthesis(audio, sr, grain_duration_ms, density, pitch_variation_semitones, output_duration_factor)`
* **Pitch Shift (Librosa-based)**: `creative_fx.apply_pitch_shift(audio, sr, n_steps)`
* **Time Stretch (Librosa-based)**: `creative_fx.apply_time_stretch(audio, sr, rate)`
* **Reverb**: `creative_fx.apply_reverb(audio, sr, reverb_time_s, decay_factor, dry_wet_mix)`
* **Chorus**: `creative_fx.apply_chorus(audio, sr, delay_ms, depth_ms, rate_hz, dry_wet_mix, num_voices)`

### Saving Audio
```python
sonifier.save_audio("output_filename.wav") # Normalizes to 16-bit by default
# sonifier.save_audio("output_filename.wav", normalize=False) # To save as-is
```

## Examples in Jupyter Notebook
A comprehensive Jupyter Notebook, `examples.ipynb`, is provided with the package. It showcases:
* Loading data.
* Using both sonification methods.
* Applying all standard and experimental effects available via `sonifier.apply_effect`.
* Demonstrating how to use functions from `additional_sound_modifiers.py` directly.
* Visualizing spectrograms.

Please refer to this notebook for practical examples and to hear the effects in action.

## Contributing
We'll figure details out later, for now, just have fun with the code! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
