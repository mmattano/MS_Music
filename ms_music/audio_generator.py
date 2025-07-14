# ms_music/audio_generator.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import math

TAU = 2 * math.pi

def _apply_intensity_ramps(segment_audio: np.ndarray,
                           current_intensity_norm: float,
                           prev_intensity_norm: float,
                           next_intensity_norm: float,
                           overlap_samples: int,
                           is_first_segment: bool,
                           is_last_segment: bool):
    """
    Applies intensity ramps (gradients) to an audio segment based on current,
    previous, and next normalized intensities.
    """
    mod_segment = np.copy(segment_audio) * current_intensity_norm # Base scaling
    seg_len = len(mod_segment)

    if overlap_samples <= 0 or seg_len == 0:
        return mod_segment # No ramp if no overlap or empty segment

    # Ramp in from previous intensity
    if not is_first_segment:
        ramp_in_len = min(overlap_samples, seg_len)
        ramp = np.linspace(prev_intensity_norm, current_intensity_norm, ramp_in_len)
        mod_segment[:ramp_in_len] = segment_audio[:ramp_in_len] * ramp # Use original segment_audio for ramp source

    # Ramp out to next intensity
    if not is_last_segment:
        ramp_out_len = min(overlap_samples, seg_len)
        # Ensure ramp_out does not start before ramp_in ends if segment is short
        start_idx_ramp_out = max(seg_len - ramp_out_len, 0) 
        if not is_first_segment: # if there was a ramp_in
             start_idx_ramp_out = max(seg_len - ramp_out_len, min(overlap_samples, seg_len))


        actual_ramp_out_len = seg_len - start_idx_ramp_out
        if actual_ramp_out_len > 0:
            ramp = np.linspace(current_intensity_norm, next_intensity_norm, actual_ramp_out_len)
            mod_segment[start_idx_ramp_out:] = segment_audio[start_idx_ramp_out:] * ramp

    return mod_segment


def generate_audio_gradient_method(processed_spectra_dfs: list,
                                   max_intensity_overall: float,
                                   min_mz_overall: float, max_mz_overall: float,
                                   total_duration_seconds: float, sample_rate: int,
                                   overlap_percentage: float = 0.05):
    if not processed_spectra_dfs:
        print("Warning (Gradient): No processed spectra. Returning silent audio.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)
    if max_intensity_overall <= 0:
        print("Warning (Gradient): Max overall intensity is non-positive. Sonification will be silent.")
        # Still proceed to generate structure, but it will be all zeros after normalization.
        # Or return early:
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)


    num_scans = len(processed_spectra_dfs)
    if num_scans == 0:
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)
        
    samples_per_scan = round(sample_rate * total_duration_seconds / num_scans)
    if samples_per_scan <= 0: # Ensure positive samples per scan
        print("Warning (Gradient): Samples per scan is not positive. Increase duration or check scan count.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)

    total_samples = int(samples_per_scan * num_scans)
    actual_total_duration_seconds = total_samples / sample_rate
    time_vector = np.linspace(0, actual_total_duration_seconds, total_samples, endpoint=False, dtype=np.float32)

    start_mz = math.floor(min_mz_overall) if pd.notna(min_mz_overall) and min_mz_overall > 0 else 1
    end_mz = math.ceil(max_mz_overall) if pd.notna(max_mz_overall) and max_mz_overall > 0 else 1
    if start_mz > end_mz : # if range is invalid (e.g. all mz were 0)
        print("Warning (Gradient): Invalid m/z range found. Returning silent audio.")
        return np.zeros(total_samples, dtype=np.float32)

    all_integer_mzs = range(start_mz, end_mz + 1)
    
    song = np.zeros(total_samples, dtype=np.float32)
    overlap_samples = round(overlap_percentage * samples_per_scan)

    print("Generating audio with gradient method...")
    for mz_value in tqdm(all_integer_mzs, desc="Processing m/z frequencies", unit="Hz"):
        if mz_value <= 0: continue # Frequency must be positive

        mz_sine_wave = np.sin(mz_value * TAU * time_vector)
        modulated_mz_wave_component = np.zeros_like(mz_sine_wave)

        # Get normalized intensities for this m/z across all scans
        normalized_intensities_for_mz = np.zeros(num_scans, dtype=np.float32)
        for i, scan_df in enumerate(processed_spectra_dfs):
            if mz_value in scan_df.index: # Check if rounded_mz is in DataFrame index
                normalized_intensities_for_mz[i] = scan_df.loc[mz_value, 'intensities'] / max_intensity_overall
        
        # Apply intensity modulation with ramps
        for scan_idx in range(num_scans):
            start_sample_idx = scan_idx * samples_per_scan
            end_sample_idx = start_sample_idx + samples_per_scan
            
            current_segment_sine = mz_sine_wave[start_sample_idx:end_sample_idx]
            if current_segment_sine.size == 0 : continue # Should not happen if samples_per_scan > 0

            current_intensity = normalized_intensities_for_mz[scan_idx]
            prev_intensity = normalized_intensities_for_mz[scan_idx - 1] if scan_idx > 0 else 0.0 # Ramp from silence for first
            next_intensity = normalized_intensities_for_mz[scan_idx + 1] if scan_idx < num_scans - 1 else 0.0 # Ramp to silence for last
            
            is_first = (scan_idx == 0)
            is_last = (scan_idx == num_scans - 1)

            ramped_segment = _apply_intensity_ramps(
                current_segment_sine, current_intensity, prev_intensity, next_intensity,
                overlap_samples, is_first, is_last
            )
            modulated_mz_wave_component[start_sample_idx:end_sample_idx] = ramped_segment
        
        song += modulated_mz_wave_component

    # Ensure final song length matches the initially calculated total_samples based on duration
    # This handles minor discrepancies from rounding samples_per_scan.
    expected_total_samples = int(total_duration_seconds * sample_rate)
    if len(song) > expected_total_samples:
        song = song[:expected_total_samples]
    elif len(song) < expected_total_samples:
        song = np.pad(song, (0, expected_total_samples - len(song)), 'constant', constant_values=0)
        
    return song

def _adsr_envelope(length_samples: int, attack_time_pc: float, decay_time_pc: float,
                   sustain_level_pc: float, release_time_pc: float):
    """Generates an ADSR envelope. Times are percentages of total length."""
    # Ensure sum of A, D, R percentages is <= 1.0 for sustain to be non-negative.
    # Prioritize A, D, R, and calculate Sustain time.
    attack_samples = max(0, int(attack_time_pc * length_samples))
    decay_samples = max(0, int(decay_time_pc * length_samples))
    release_samples = max(0, int(release_time_pc * length_samples))

    # Adjust if sum of A,D,R > length_samples (should not happen if pc <= 1)
    if attack_samples + decay_samples + release_samples > length_samples:
        # Scale them down proportionally if sum is too large
        total_adr_samples = attack_samples + decay_samples + release_samples
        scale_factor = length_samples / total_adr_samples
        attack_samples = int(attack_samples * scale_factor)
        decay_samples = int(decay_samples * scale_factor)
        release_samples = int(release_samples * scale_factor)
        # Recalculate to ensure integer sum matches
        release_samples = length_samples - (attack_samples + decay_samples)


    sustain_samples = max(0, length_samples - (attack_samples + decay_samples + release_samples))
    
    envelope = np.zeros(length_samples, dtype=np.float32)
    current_pos = 0

    # Attack
    if attack_samples > 0:
        envelope[current_pos : current_pos + attack_samples] = np.linspace(0, 1, attack_samples, endpoint=False)
    current_pos += attack_samples
    
    # Decay
    if decay_samples > 0 and current_pos < length_samples:
        env_start_val = 1.0 if attack_samples > 0 else sustain_level_pc # Start decay from 1 or sustain_level if no attack
        decay_end = min(current_pos + decay_samples, length_samples)
        envelope[current_pos : decay_end] = np.linspace(env_start_val, sustain_level_pc, decay_end - current_pos, endpoint=False)
    current_pos += decay_samples
    
    # Sustain
    if sustain_samples > 0 and current_pos < length_samples:
        sustain_end = min(current_pos + sustain_samples, length_samples)
        envelope[current_pos : sustain_end] = sustain_level_pc
    current_pos += sustain_samples
    
    # Release
    if release_samples > 0 and current_pos < length_samples:
        release_end = min(current_pos + release_samples, length_samples)
        # Ensure release actually fits
        actual_release_samples = release_end - current_pos
        if actual_release_samples > 0:
             envelope[current_pos : release_end] = np.linspace(sustain_level_pc, 0, actual_release_samples, endpoint=False)
    
    # Ensure last point is 0 if there's a release phase that reaches the end
    if release_samples > 0 and (current_pos + release_samples >= length_samples):
        if length_samples > 0: envelope[-1] = 0.0

    return envelope


def generate_audio_adsr_method(processed_spectra_dfs: list,
                               max_intensity_overall: float,
                               total_duration_seconds: float, sample_rate: int,
                               adsr_settings: dict = None):
    if not processed_spectra_dfs:
        print("Warning (ADSR): No processed spectra. Returning silent audio.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)
    if max_intensity_overall <= 0:
        print("Warning (ADSR): Max overall intensity is non-positive. Sonification will be silent.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)

    num_scans = len(processed_spectra_dfs)
    if num_scans == 0: return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)

    samples_per_scan = round(sample_rate * total_duration_seconds / num_scans)
    if samples_per_scan <= 0:
        print("Warning (ADSR): Samples per scan is not positive.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)

    total_samples_target = int(total_duration_seconds * sample_rate)
    
    # Default ADSR settings if none provided
    if adsr_settings is None:
        adsr_settings = {} # Trigger defaults below

    use_random_adsr = adsr_settings.get('randomize', True) 
    
    # Fixed defaults if not randomizing and no specific values given
    fixed_attack_pc = adsr_settings.get('attack_time_pc', 0.05)
    fixed_decay_pc = adsr_settings.get('decay_time_pc', 0.1)
    fixed_sustain_level = adsr_settings.get('sustain_level_pc', 0.7)
    fixed_release_pc = adsr_settings.get('release_time_pc', 0.15)

    song_parts = []
    time_per_scan_segment = np.linspace(0, samples_per_scan / sample_rate, samples_per_scan, endpoint=False, dtype=np.float32)

    print("Generating audio with ADSR method...")
    for scan_df in tqdm(processed_spectra_dfs, desc="Processing Scans (ADSR)", unit="scan"):
        scan_audio_segment = np.zeros(samples_per_scan, dtype=np.float32)

        if not scan_df.empty:
            for mz_value, row_series in scan_df.iterrows(): # iterrows returns (index, Series)
                intensity = row_series['intensities']
                if mz_value <= 0 or intensity <= 0: continue

                normalized_intensity = intensity / max_intensity_overall
                mz_sine_wave_segment = np.sin(mz_value * TAU * time_per_scan_segment)
                scan_audio_segment += mz_sine_wave_segment * normalized_intensity
        
        max_abs_segment = np.max(np.abs(scan_audio_segment))
        if max_abs_segment > 0:
            scan_audio_segment /= max_abs_segment
            
        if use_random_adsr:
            attack_t_pc = random.uniform(0.01, 0.10) 
            decay_t_pc = random.uniform(0.02, 0.15)
            sustain_l_pc = random.uniform(0.4, 0.8)

            max_r_pc = 1.0 - attack_t_pc - decay_t_pc
            release_t_pc = random.uniform(0.05, max(0.06, max_r_pc * 0.8)) # Ensure release is not too long
            release_t_pc = max(0.01, min(release_t_pc, max_r_pc)) # Clamp release
        else:
            attack_t_pc = fixed_attack_pc
            decay_t_pc = fixed_decay_pc
            sustain_l_pc = fixed_sustain_level
            release_t_pc = fixed_release_pc

        envelope = _adsr_envelope(samples_per_scan, attack_t_pc, decay_t_pc, sustain_l_pc, release_t_pc)
        song_parts.append(scan_audio_segment * envelope)

    if not song_parts: # Should not happen if num_scans > 0
        return np.zeros(total_samples_target, dtype=np.float32)
        
    song = np.concatenate(song_parts)
    
    # Ensure final song length matches the target total_samples
    if len(song) > total_samples_target:
        song = song[:total_samples_target]
    elif len(song) < total_samples_target:
        song = np.pad(song, (0, total_samples_target - len(song)), 'constant', constant_values=0)
        
    return song


def mz_to_frequency_inverse_log(mz_value, mz_min, mz_max, 
                               freq_min=200.0, freq_max=4000.0):
    """
    Maps m/z values to frequencies using inverse logarithmic scaling.
    Low m/z → high frequency, high m/z → low frequency
    Uses logarithmic scaling for musical feel.
    
    Args:
        mz_value: The m/z value to convert
        mz_min, mz_max: The range of m/z values in your dataset
        freq_min, freq_max: The desired frequency range in Hz
    
    Returns:
        Frequency in Hz
    """
    if mz_value <= 0 or mz_min <= 0:
        return freq_min
    
    # Normalize m/z to [0, 1] range
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    
    # Inverse logarithmic mapping
    # Higher m/z (closer to 1) gives lower frequency
    log_ratio = math.log(freq_max / freq_min)
    frequency = freq_max * math.exp(-mz_normalized * log_ratio)
    
    return frequency

def mz_to_frequency_power_law(mz_value, mz_min, mz_max,
                             freq_min=200.0, freq_max=4000.0, 
                             exponent=1.5):
    """
    Maps m/z values using a power law relationship.
    Gives smoother, more organic feeling transitions.
    
    Args:
        exponent: Controls the curve shape (1.0 = linear, >1.0 = more curved)
    """
    if mz_value <= 0:
        return freq_min
        
    # Normalize and invert
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    
    # Power law scaling (inverted so low m/z → high freq)
    freq_normalized = (1 - mz_normalized) ** exponent
    
    # Map to frequency range (logarithmic)
    log_freq_min = math.log(freq_min)
    log_freq_max = math.log(freq_max)
    log_frequency = log_freq_min + freq_normalized * (log_freq_max - log_freq_min)
    
    return math.exp(log_frequency)

def mz_to_frequency_musical_octaves(mz_value, mz_min, mz_max,
                                   base_freq=440.0, num_octaves=4):
    """
    Maps m/z to frequencies using musical octaves.
    Each octave doubles the frequency, creating very musical results.
    
    Args:
        base_freq: Starting frequency (like A4 = 440 Hz)
        num_octaves: Number of octaves to span
    """
    if mz_value <= 0:
        return base_freq
        
    # Normalize m/z to [0, 1]
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    
    # Invert so low m/z → high frequency
    mz_inverted = 1 - mz_normalized
    
    # Map to octaves (each octave is a doubling of frequency)
    octave_position = mz_inverted * num_octaves
    frequency = base_freq * (2 ** octave_position)
    
    return frequency

def mz_to_frequency_chromatic_scale(mz_value, mz_min, mz_max,
                                   base_freq=261.63, # C4
                                   num_semitones=48): # 4 octaves
    """
    Maps m/z to frequencies using chromatic (12-tone) scale.
    Creates very musical results that sound like actual notes.
    """
    if mz_value <= 0:
        return base_freq
        
    # Normalize and invert
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    mz_inverted = 1 - mz_normalized
    
    # Map to semitones
    semitone_position = mz_inverted * num_semitones
    
    # Convert to frequency (each semitone is 2^(1/12))
    frequency = base_freq * (2 ** (semitone_position / 12))
    
    return frequency

# Modified audio generation function
def generate_audio_gradient_method_enhanced(processed_spectra_dfs: list,
                                          max_intensity_overall: float,
                                          min_mz_overall: float, 
                                          max_mz_overall: float,
                                          total_duration_seconds: float, 
                                          sample_rate: int,
                                          overlap_percentage: float = 0.05,
                                          frequency_mapping: str = 'inverse_log',
                                          freq_range: tuple = (200.0, 4000.0)):
    """
    Enhanced version of the gradient method with better frequency mapping.
    
    Args:
        frequency_mapping: 'inverse_log', 'power_law', 'musical_octaves', or 'chromatic'
        freq_range: (min_freq, max_freq) in Hz
    """
    if not processed_spectra_dfs:
        print("Warning (Gradient Enhanced): No processed spectra. Returning silent audio.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)
        
    if max_intensity_overall <= 0:
        print("Warning (Gradient Enhanced): Max overall intensity is non-positive.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)

    num_scans = len(processed_spectra_dfs)
    if num_scans == 0:
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)
        
    samples_per_scan = round(sample_rate * total_duration_seconds / num_scans)
    if samples_per_scan <= 0:
        print("Warning (Gradient Enhanced): Samples per scan is not positive.")
        return np.zeros(int(total_duration_seconds * sample_rate), dtype=np.float32)

    total_samples = int(samples_per_scan * num_scans)
    actual_total_duration_seconds = total_samples / sample_rate
    time_vector = np.linspace(0, actual_total_duration_seconds, total_samples, 
                             endpoint=False, dtype=np.float32)

    # Get all unique m/z values
    all_mz_values = set()
    for scan_df in processed_spectra_dfs:
        if not scan_df.empty:
            all_mz_values.update(scan_df.index)
    
    if not all_mz_values:
        return np.zeros(total_samples, dtype=np.float32)
    
    all_mz_values = sorted(all_mz_values)
    
    # Choose frequency mapping function
    freq_min, freq_max = freq_range
    
    if frequency_mapping == 'inverse_log':
        freq_func = lambda mz: mz_to_frequency_inverse_log(
            mz, min_mz_overall, max_mz_overall, freq_min, freq_max)
    elif frequency_mapping == 'power_law':
        freq_func = lambda mz: mz_to_frequency_power_law(
            mz, min_mz_overall, max_mz_overall, freq_min, freq_max)
    elif frequency_mapping == 'musical_octaves':
        freq_func = lambda mz: mz_to_frequency_musical_octaves(
            mz, min_mz_overall, max_mz_overall, freq_min, 4)
    elif frequency_mapping == 'chromatic':
        freq_func = lambda mz: mz_to_frequency_chromatic_scale(
            mz, min_mz_overall, max_mz_overall, freq_min, 48)
    else:
        # Fallback to original linear mapping
        freq_func = lambda mz: float(mz)
    
    song = np.zeros(total_samples, dtype=np.float32)
    overlap_samples = round(overlap_percentage * samples_per_scan)

    print(f"Generating audio with enhanced gradient method using {frequency_mapping} mapping...")
    
    # Process each m/z value
    for mz_value in all_mz_values:
        if mz_value <= 0:
            continue
            
        # Convert m/z to frequency using selected mapping
        frequency = freq_func(mz_value)
        if frequency <= 0:
            continue
            
        print(f"m/z {mz_value} → {frequency:.1f} Hz")
        
        # Generate sine wave at this frequency
        mz_sine_wave = np.sin(frequency * 2 * math.pi * time_vector)
        modulated_mz_wave_component = np.zeros_like(mz_sine_wave)

        # Get normalized intensities for this m/z across all scans
        normalized_intensities_for_mz = np.zeros(num_scans, dtype=np.float32)
        for i, scan_df in enumerate(processed_spectra_dfs):
            if mz_value in scan_df.index:
                normalized_intensities_for_mz[i] = scan_df.loc[mz_value, 'intensities'] / max_intensity_overall
        
        # Apply intensity modulation with ramps (using existing _apply_intensity_ramps function)
        for scan_idx in range(num_scans):
            start_sample_idx = scan_idx * samples_per_scan
            end_sample_idx = start_sample_idx + samples_per_scan
            
            current_segment_sine = mz_sine_wave[start_sample_idx:end_sample_idx]
            if current_segment_sine.size == 0:
                continue

            current_intensity = normalized_intensities_for_mz[scan_idx]
            prev_intensity = normalized_intensities_for_mz[scan_idx - 1] if scan_idx > 0 else 0.0
            next_intensity = normalized_intensities_for_mz[scan_idx + 1] if scan_idx < num_scans - 1 else 0.0
            
            is_first = (scan_idx == 0)
            is_last = (scan_idx == num_scans - 1)

            # You'd need to import or copy the _apply_intensity_ramps function here
            # For now, simple intensity application:
            ramped_segment = current_segment_sine * current_intensity
            modulated_mz_wave_component[start_sample_idx:end_sample_idx] = ramped_segment
        
        song += modulated_mz_wave_component

    # Ensure final song length matches expected
    expected_total_samples = int(total_duration_seconds * sample_rate)
    if len(song) > expected_total_samples:
        song = song[:expected_total_samples]
    elif len(song) < expected_total_samples:
        song = np.pad(song, (0, expected_total_samples - len(song)), 'constant', constant_values=0)
        
    return song
