# additional_sound_modifiers.py

import numpy as np
import librosa
from scipy import signal
import os

TAU = 2 * np.pi

def apply_granular_synthesis(audio_data: np.ndarray, sample_rate: int,
                             grain_duration_ms: float = 50,
                             density: float = 1.0, # Relative density of grains
                             pitch_variation_semitones: float = 0.0, # Max pitch shift in semitones
                             output_duration_factor: float = 1.0 # Factor to scale output duration relative to input
                             ):
    """
    Applies a granular synthesis effect with pitch variation.
    Chops audio into small grains, potentially pitch-shifts them, and re-synthesizes.

    Args:
        audio_data (np.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the audio.
        grain_duration_ms (float): Duration of each grain in milliseconds.
        density (float): Multiplier for the number of grains. If > 1, more grains than
                         would simply tile the input are generated, leading to denser textures.
        pitch_variation_semitones (float): Max random pitch shift per grain in semitones
                                           (e.g., 2.0 for +/- 2 semitones). Set to 0 for no variation.
        output_duration_factor (float): Multiplies the input audio duration to set the output duration.
                                        Allows for stretching or compressing the granular texture in time.
    Returns:
        np.ndarray: Granulated audio signal (mono).
    """
    print(f"Applying Granular Synthesis: GrainDur={grain_duration_ms}ms, Density={density}, "
          f"PitchVar=+/-{pitch_variation_semitones}st, OutDurFactor={output_duration_factor}")

    if audio_data.size == 0:
        print("Warning (Granular): Input audio is empty. Skipping.")
        return np.array([], dtype=np.float32)

    audio_float = audio_data.astype(np.float32)
    if audio_float.ndim > 1:
        audio_float = librosa.to_mono(audio_float)

    grain_length_samples = int((grain_duration_ms / 1000.0) * sample_rate)
    if grain_length_samples <= 1 or len(audio_float) < grain_length_samples // 2 : # Need some audio to granulate
        print("Warning (Granular): Audio too short for grain size or grain size too small. Skipping.")
        return audio_float

    # Calculate number of grains: related to original duration, grain length, and density
    num_base_grains = len(audio_float) / grain_length_samples
    num_grains_to_generate = int(num_base_grains * density)
    if num_grains_to_generate <= 0: num_grains_to_generate = 10 # Generate at least a few grains

    output_length = int(len(audio_float) * output_duration_factor)
    if output_length <=0: output_length = len(audio_float) # Fallback
    output_audio = np.zeros(output_length, dtype=np.float32)
    
    grain_window = signal.windows.hann(grain_length_samples, sym=False) # Use sym=False for OLA

    for _ in range(num_grains_to_generate):
        # Source position for grain
        max_start_pos_source = len(audio_float) - grain_length_samples
        if max_start_pos_source < 0: max_start_pos_source = 0 # If audio shorter than grain
        start_pos_source = np.random.randint(0, max_start_pos_source + 1)
        grain = np.copy(audio_float[start_pos_source : start_pos_source + grain_length_samples])

        if grain.size < grain_length_samples: # Pad if extracted grain is too short (end of audio)
            grain = np.pad(grain, (0, grain_length_samples - len(grain)), 'constant')

        # Pitch variation
        if pitch_variation_semitones != 0.0:
            shift_semitones = np.random.uniform(-pitch_variation_semitones, pitch_variation_semitones)
            if abs(shift_semitones) > 1e-3: # Apply only if shift is significant
                try:
                    # Pitch shifting can change length slightly, ensure it's handled
                    grain_shifted = librosa.effects.pitch_shift(y=grain, sr=sample_rate, n_steps=shift_semitones)
                    if len(grain_shifted) == grain_length_samples:
                        grain = grain_shifted
                    elif len(grain_shifted) > grain_length_samples:
                        grain = grain_shifted[:grain_length_samples] # Truncate
                    else: # Pad
                        grain = np.pad(grain_shifted, (0, grain_length_samples - len(grain_shifted)), 'constant')
                except Exception as e:
                    print(f"Warning (Granular Pitch Shift): {e}. Using original grain pitch.")
        
        grain *= grain_window # Apply window after potential pitch shift

        # Output position for grain (overlap-add)
        max_start_pos_output = output_length - grain_length_samples
        if max_start_pos_output < 0 : max_start_pos_output = 0 # If output shorter than grain
        start_pos_output = np.random.randint(0, max_start_pos_output + 1)
        
        # Ensure indices are within bounds for addition
        end_idx_output = start_pos_output + grain_length_samples
        if end_idx_output <= output_length:
            output_audio[start_pos_output : end_idx_output] += grain
        else: # Grain goes past end of output buffer, add what fits
            fit_length = output_length - start_pos_output
            if fit_length > 0:
                output_audio[start_pos_output:] += grain[:fit_length]


    max_abs_out = np.max(np.abs(output_audio))
    if max_abs_out > 1e-9: # Avoid division by zero/small numbers
        output_audio /= max_abs_out # Normalize
        
    return output_audio


def apply_pitch_shift(audio_data: np.ndarray, sample_rate: int, n_steps: float):
    print(f"Applying Pitch Shift (Steps: {n_steps})")
    if audio_data.size == 0: return np.copy(audio_data)
    return librosa.effects.pitch_shift(y=audio_data.astype(np.float32), sr=sample_rate, n_steps=n_steps)

def apply_time_stretch(audio_data: np.ndarray, sample_rate: int, rate: float):
    print(f"Applying Time Stretch (Rate: {rate})")
    if audio_data.size == 0: return np.copy(audio_data)
    if rate <= 0:
        print("Warning (Time Stretch): Rate must be positive. Skipping.")
        return np.copy(audio_data)
    return librosa.effects.time_stretch(y=audio_data.astype(np.float32), rate=rate)

def apply_reverb(audio_data: np.ndarray, sample_rate: int,
                 reverb_time_s: float = 0.7, # Default T60
                 decay_factor: float = 6.908, # ln(1000) for T60 definition
                 dry_wet_mix: float = 0.3):
    print(f"Applying Reverb (T60: {reverb_time_s}s, Mix: {dry_wet_mix})")
    if audio_data.size == 0: return np.copy(audio_data)

    audio_float = audio_data.astype(np.float32)
    if audio_float.ndim > 1:
        audio_float = librosa.to_mono(audio_float)

    if reverb_time_s <= 0:
        print("Warning (Reverb): Reverb time must be positive. Returning dry signal.")
        return audio_float

    impulse_len_samples = int(reverb_time_s * sample_rate)
    if impulse_len_samples == 0: return audio_float

    noise_impulse = np.random.randn(impulse_len_samples).astype(np.float32)
    time_points = np.arange(impulse_len_samples) / sample_rate
    
    # Correct T60 decay: e^(-decay_factor * t / T60)
    # decay_factor = ln(1000) makes T60 the time to decay by 60dB
    if reverb_time_s < 1e-6 : reverb_time_s = 1e-6 # Avoid division by zero
    decay_envelope = np.exp(-decay_factor * time_points / reverb_time_s)
    impulse_response = noise_impulse * decay_envelope
    
    max_ir_abs = np.max(np.abs(impulse_response))
    if max_ir_abs > 1e-9: impulse_response /= max_ir_abs # Normalize IR peak

    wet_signal = signal.convolve(audio_float, impulse_response, mode='same')

    max_abs_dry = np.max(np.abs(audio_float))
    max_abs_wet = np.max(np.abs(wet_signal))
    if max_abs_wet > 1e-9 and max_abs_dry > 1e-9:
        wet_signal *= (max_abs_dry / max_abs_wet) # Scale wet peak to dry peak
    elif max_abs_wet > 1e-9: # Dry is silent
         wet_signal /= max_abs_wet # Normalize wet to [-1,1]

    return (1.0 - dry_wet_mix) * audio_float + dry_wet_mix * wet_signal

def apply_chorus(audio_data: np.ndarray, sample_rate: int,
                 delay_ms: float = 20.0, depth_ms: float = 2.0, rate_hz: float = 0.7,
                 dry_wet_mix: float = 0.5, feedback: float = 0.2, num_voices: int = 1):
    print(f"Applying Chorus: Delay={delay_ms}ms, Depth={depth_ms}ms, Rate={rate_hz}Hz, Mix={dry_wet_mix}, Voices={num_voices}")
    if audio_data.size == 0: return np.copy(audio_data)
    audio_float = audio_data.astype(np.float32)
    
    delay_samples_base = int(delay_ms / 1000.0 * sample_rate)
    depth_samples = int(depth_ms / 1000.0 * sample_rate)
    feedback = np.clip(feedback, 0, 0.9) # Keep feedback stable

    wet_signal_total = np.zeros_like(audio_float)

    for voice in range(num_voices):
        # Slightly different LFO phase for each voice for richer chorus
        lfo_phase_offset = (TAU * voice) / num_voices if num_voices > 0 else 0
        t = np.arange(len(audio_float)) / sample_rate
        lfo = depth_samples * np.sin(TAU * rate_hz * t + lfo_phase_offset)
        
        current_delay_line = np.zeros(len(audio_float) + delay_samples_base + depth_samples + 5, dtype=np.float32) # Max delay buffer
        current_wet_voice = np.zeros_like(audio_float)

        # Simplified delay implementation using array indexing + interpolation
        # A more efficient real-time approach would use circular buffers.
        for i in range(len(audio_float)):
            # Signal to be written into delay line for this voice
            # For feedback, this should be audio_float[i] + feedback * past_delayed_output_of_this_voice
            # For simplicity, let's use feedback from the input signal for now (simpler flanger-like)
            # or omit direct feedback in this simplified loop.
            # For a basic chorus (no feedback in this direct path):
            input_to_delay = audio_float[i]
            
            # Current modulated delay for this voice
            modulated_delay_samps = delay_samples_base + lfo[i]
            modulated_delay_samps = np.clip(modulated_delay_samps, 0, current_delay_line.shape[0] - 1.01) # ensure idx_ceil is in bounds

            read_idx_exact = i - modulated_delay_samps # Where to read from the original signal

            delayed_sample_val = 0.0
            if read_idx_exact >= 0:
                idx_floor = int(np.floor(read_idx_exact))
                idx_ceil = int(np.ceil(read_idx_exact))
                frac = read_idx_exact - idx_floor

                sample_at_floor = audio_float[idx_floor] if idx_floor < len(audio_float) else 0.0
                sample_at_ceil = audio_float[idx_ceil] if idx_ceil < len(audio_float) else 0.0
                
                delayed_sample_val = (1 - frac) * sample_at_floor + frac * sample_at_ceil
            
            current_wet_voice[i] = delayed_sample_val
        
        wet_signal_total += current_wet_voice

    if num_voices > 0:
        wet_signal_total /= num_voices # Average voices to prevent excessive loudness

    max_abs_wet = np.max(np.abs(wet_signal_total))
    if max_abs_wet > 1e-9:
        max_abs_dry = np.max(np.abs(audio_float)) if np.max(np.abs(audio_float)) > 1e-9 else 1.0
        wet_signal_total = wet_signal_total / max_abs_wet * max_abs_dry # Scale wet to dry peak

    return (1.0 - dry_wet_mix) * audio_float + dry_wet_mix * wet_signal_total
