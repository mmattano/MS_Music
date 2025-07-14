# ms_music/effects.py

import numpy as np
import librosa
from scipy import signal

TAU = 2 * np.pi

def apply_hpss(audio_data: np.ndarray, sample_rate: int, margin: float = 16.0,
               harmonic: bool = True, percussive: bool = False) -> np.ndarray:
    if not harmonic and not percussive:
        print("Warning (HPSS): Both harmonic and percussive are False. Returning original audio.")
        return np.copy(audio_data)
    if audio_data.size == 0: return np.copy(audio_data)
    
    print(f"Applying HPSS: Margin={margin}, Harmonic Output={harmonic}, Percussive Output={percussive}")
    audio_float = audio_data.astype(np.float32)
    D = librosa.stft(audio_float)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)

    output_audio = np.zeros_like(audio_float)
    if harmonic:
        output_audio += librosa.istft(D_harmonic, length=len(audio_float))
    if percussive:
        output_audio += librosa.istft(D_percussive, length=len(audio_float))
    return output_audio

def apply_notch_filter(audio_data: np.ndarray, sample_rate: int, notch_freq: float,
                       quality_factor: float = 10.0) -> np.ndarray:
    if audio_data.size == 0: return np.copy(audio_data)
    if not (0 < notch_freq < sample_rate / 2):
        print(f"Warning (Notch Filter): Invalid notch_freq {notch_freq} Hz. Skipping.")
        return np.copy(audio_data)
    
    print(f"Applying Notch Filter: Frequency={notch_freq} Hz, Q={quality_factor}")
    b, a = signal.iirnotch(notch_freq, quality_factor, fs=sample_rate)
    return signal.filtfilt(b, a, audio_data)

def apply_butterworth_filter(audio_data: np.ndarray, sample_rate: int, cutoff_freq,
                             btype: str = 'low', order: int = 4,
                             gustafson_method: bool = False) -> np.ndarray:
    if audio_data.size == 0: return np.copy(audio_data)
    nyquist = 0.5 * sample_rate
    normalized_cutoff = np.array(cutoff_freq) / nyquist
    
    if np.any(normalized_cutoff <= 0) or np.any(normalized_cutoff >= 1):
        print(f"Warning (Butterworth): Invalid cutoff_freq {cutoff_freq} Hz. Skipping.")
        return np.copy(audio_data)

    print(f"Applying Butterworth Filter: Type={btype}, Cutoff(s)={cutoff_freq} Hz, Order={order}, Gustafson={gustafson_method}")
    b, a = signal.butter(order, normalized_cutoff, btype=btype, analog=False)
    
    filtfilt_method = "gust" if gustafson_method else "pad"
    try:
        return signal.filtfilt(b, a, audio_data, method=filtfilt_method)
    except ValueError: # Fallback for Gustafson if it fails (e.g., very short signals)
        print(f"Warning (Butterworth): filtfilt method='{filtfilt_method}' failed. Using default 'pad'.")
        return signal.filtfilt(b, a, audio_data, method="pad")

def apply_chebyshev1_filter(audio_data: np.ndarray, sample_rate: int, cutoff_freq, 
                            ripple_db: float, btype: str = 'low', order: int = 4) -> np.ndarray:
    if audio_data.size == 0: return np.copy(audio_data)
    nyquist = 0.5 * sample_rate
    normalized_cutoff = np.array(cutoff_freq) / nyquist

    if np.any(normalized_cutoff <= 0) or np.any(normalized_cutoff >= 1):
        print(f"Warning (Chebyshev1): Invalid cutoff_freq {cutoff_freq} Hz. Skipping.")
        return np.copy(audio_data)
    if ripple_db <= 0:
        print(f"Warning (Chebyshev1): ripple_db must be positive. Given: {ripple_db}. Skipping.")
        return np.copy(audio_data)

    print(f"Applying Chebyshev Type 1 Filter: Type={btype}, Cutoff(s)={cutoff_freq} Hz, Ripple={ripple_db} dB, Order={order}")
    b, a = signal.cheby1(order, ripple_db, normalized_cutoff, btype=btype, analog=False)
    return signal.filtfilt(b, a, audio_data)

# EXPERIMENTAL: These effects are not yet fully tested and may change in future versions.

def apply_experimental_pll_filter(audio_data: np.ndarray, sample_rate: int,
                                  center_freq: float = 500.0,
                                  loop_bandwidth_hz: float = 20.0,
                                  damping_factor: float = 0.707,
                                  output_filter_freq_offset_hz: float = 0.0,
                                  output_filter_min_freq_hz: float = 20.0,
                                  output_filter_max_freq_hz: float = None
                                  ) -> np.ndarray:
    """
    EXPERIMENTAL: Phase-Locked Loop (PLL) based time-varying filter.
    This implementation uses a PLL to estimate the instantaneous frequency
    of the input signal. This estimated frequency then controls the cutoff
    of a 1st-order low-pass filter applied to the original audio.

    Args:
        audio_data (np.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the audio (Hz).
        center_freq (float): Initial center frequency of the NCO and quiescent
                             frequency of the output filter (Hz).
        loop_bandwidth_hz (float): Approximate bandwidth of the PLL loop (Hz).
                                   Controls responsiveness. Lower is slower/smoother.
        damping_factor (float): Damping factor for the PLL's second-order loop filter.
                                Typically 0.707 for critical damping.
        output_filter_freq_offset_hz (float): An offset added to the NCO's instantaneous
                                              frequency to set the output filter's cutoff.
                                              Can be positive or negative.
        output_filter_min_freq_hz (float): Minimum cutoff frequency for the output LPF.
        output_filter_max_freq_hz (float, optional): Maximum cutoff frequency for the output LPF.
                                                     Defaults to sample_rate / 2.2 to keep well below Nyquist.

    Returns:
        np.ndarray: The processed audio signal.
    """
    print(f"Applying EXPERIMENTAL PLL Time-Varying Filter: CenterFreq={center_freq} Hz, LoopBW={loop_bandwidth_hz} Hz")
    if audio_data.size == 0:
        return np.copy(audio_data)

    # Ensure audio data is float for processing
    audio_float = audio_data.astype(np.float32)


    if output_filter_max_freq_hz is None:
        output_filter_max_freq_hz = sample_rate / 2.2 # Keep well below Nyquist
    output_filter_min_freq_hz = max(1.0, output_filter_min_freq_hz) # Ensure positive

    # PLL Parameters (Type II for zero steady-state phase error to freq step)
    omega_n_norm = loop_bandwidth_hz * TWO_PI / sample_rate  # Normalized natural frequency
    # Loop filter gains (proportional and integral paths)
    kp = 2 * damping_factor * omega_n_norm # Proportional gain (K_1 in some texts)
    ki = omega_n_norm**2                   # Integral gain (K_2 in some texts)

    nco_phase = 0.0
    # Initial NCO frequency matches center_freq
    nco_freq_rad_per_sample_center = center_freq * TWO_PI / sample_rate
    
    integrator_state = 0.0 # Loop filter integrator
    output_filter_state = 0.0 # Output 1st order LPF state
    output_audio = np.zeros_like(audio_float, dtype=np.float32)

    for i in range(len(audio_float)):
        input_sample = audio_float[i]

        # NCO quadrature output for phase detection
        nco_ref_quad = np.cos(nco_phase) # Using cos(phase) as the reference

        # Phase Detector (PD) error: input * reference_quadrature
        # This is a common PD type. Assumes input is somewhat normalized or PD gain handles amplitude.
        pd_error = input_sample * nco_ref_quad

        # Loop Filter (PI controller)
        integrator_state += ki * pd_error # Accumulate error for integral action
        loop_filter_output = kp * pd_error + integrator_state # Control signal for NCO frequency adjustment

        # NCO Frequency Update: LFO output is deviation from center in rad/sample
        current_nco_freq_rad_per_sample = nco_freq_rad_per_sample_center + loop_filter_output
        
        # Clamp NCO frequency to avoid instability / aliasing
        # Allow NCO to track slightly outside the output filter's final clamped range
        min_nco_rad_samp = (output_filter_min_freq_hz * 0.1) * TWO_PI / sample_rate # Allow NCO to go lower
        max_nco_rad_samp = (output_filter_max_freq_hz * 2.0) * TWO_PI / sample_rate # Allow NCO to go higher
        min_nco_rad_samp = max(min_nco_rad_samp, 0.001 * TWO_PI / sample_rate) # Ensure positive
        max_nco_rad_samp = min(max_nco_rad_samp, (sample_rate / 2.05) * TWO_PI / sample_rate) # Hard limit near Nyquist for NCO

        current_nco_freq_rad_per_sample = np.clip(current_nco_freq_rad_per_sample,
                                                  min_nco_rad_samp,
                                                  max_nco_rad_samp)
        
        # NCO Phase Accumulation
        nco_phase += current_nco_freq_rad_per_sample
        nco_phase %= TWO_PI # Wrap phase to [0, 2*pi)

        # NCO's instantaneous frequency (estimated input frequency)
        nco_instantaneous_freq_hz = current_nco_freq_rad_per_sample * sample_rate / TWO_PI
        
        # Determine the cutoff for the output LPF
        output_lpf_cutoff_hz = nco_instantaneous_freq_hz + output_filter_freq_offset_hz
        output_lpf_cutoff_hz = np.clip(output_lpf_cutoff_hz,
                                       output_filter_min_freq_hz,
                                       output_filter_max_freq_hz)

        # Calculate 1st order LPF coefficient 'a_coeff' for y[n] = (1-a_coeff)*y[n-1] + a_coeff*x[n]
        # Using the form a_coeff = 1 - exp(-TWO_PI * fc / fs) which comes from pole placement (z = e^(-sT))
        # This is generally more stable and accurate for varying cutoffs than simpler approximations.
        if output_lpf_cutoff_hz <= 1e-3 : # Avoid math domain error for log/exp if cutoff is effectively zero
            a_coeff = 0.0 # No filtering, effectively pass previous state or zero if input is zero
        else:
            a_coeff = 1.0 - np.exp(-TWO_PI * output_lpf_cutoff_hz / sample_rate)
        
        a_coeff = np.clip(a_coeff, 0.0, 1.0) # Ensure coefficient is valid [0, 1]

        # Apply the 1st order LPF
        output_filter_state = (1.0 - a_coeff) * output_filter_state + a_coeff * input_sample
        output_audio[i] = output_filter_state
        
    return output_audio


def apply_experimental_lms_modulation(audio_data: np.ndarray, sample_rate: int,
                                      n_segments: int = 100, mu: float = 0.05) -> np.ndarray:
    print(f"Applying EXPERIMENTAL LMS Modulation (Segments={n_segments}, Mu={mu})...")
    if audio_data.size == 0: return np.copy(audio_data)
    wave = np.copy(audio_data).astype(np.float32) # Ensure float

    if n_segments <= 0:
        print("Warning (LMS): n_segments must be positive. Skipping.")
        return wave
    segment_length = len(wave) // n_segments
    if segment_length == 0:
        print("Warning (LMS): segment_length is zero. Skipping.")
        return wave

    mod_wave = np.zeros_like(wave, dtype=np.float32)
    filter_coeffs = np.zeros(segment_length, dtype=np.float32)
    target_freq_component = sample_rate / 4.0

    for i in range(n_segments):
        start_idx = i * segment_length
        # Define current segment, potentially padding the last one
        if i == n_segments - 1: # Last segment might be shorter
            segment_raw = wave[start_idx:]
            current_segment_len = len(segment_raw)
            if current_segment_len == 0: continue
            segment_processed = np.pad(segment_raw, (0, segment_length - current_segment_len), 'constant')
        else:
            segment_processed = wave[start_idx : start_idx + segment_length]
            current_segment_len = segment_length
        
        windowed_segment = segment_processed * np.hanning(segment_length)
        dot_product_input = np.flip(windowed_segment)
        estimated_component = np.dot(np.flip(filter_coeffs), dot_product_input)
        error = target_freq_component - estimated_component

        prev_mod_val = mod_wave[start_idx - 1] if start_idx > 0 else 0.0
        mod_increment = mu * error * np.flip(windowed_segment)
        
        mod_wave_segment_update = prev_mod_val + mod_increment
        mod_wave[start_idx : start_idx + current_segment_len] = mod_wave_segment_update[:current_segment_len]
        filter_coeffs += mu * error * np.flip(windowed_segment)

    # Heuristic clipping for stability of mod_wave before np.sin
    mod_wave_abs_max = np.max(np.abs(mod_wave))
    # Choose a large but finite clip value related to number of cycles over signal.
    # If mod_wave represents phase, typical values might be in hundreds or thousands of radians.
    # If it represents frequency, it could be large.
    # Based on np.sin(2*pi*mod_wave), mod_wave is phase-like. Clip to e.g. +/- 1000*2*pi.
    # This prevents extreme frequencies from sin.
    clip_limit = 1000 * TAU 
    if mod_wave_abs_max > clip_limit:
        print(f"Warning (LMS): mod_wave reached large values (max abs: {mod_wave_abs_max:.2e}). Clipping to +/-{clip_limit:.2e}.")
        mod_wave = np.clip(mod_wave, -clip_limit, clip_limit)
    
    return np.sin(mod_wave) * wave # mod_wave here is a phase modulator


def apply_experimental_spectral_analysis_fm(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    print("Applying EXPERIMENTAL Spectral Analysis FM...")
    if audio_data.size == 0: return np.copy(audio_data)
    song = np.copy(audio_data).astype(np.float32)
    FS = sample_rate
    num_samples = len(song)

    nperseg_welch = min(num_samples, max(256, FS // 40)) # Shorter nperseg for better time resolution if needed
    if nperseg_welch <= 0 or nperseg_welch > num_samples: nperseg_welch = num_samples
    
    frequencies, spectrum = signal.welch(song, FS, nperseg=nperseg_welch, noverlap=max(0,nperseg_welch//2))

    if len(frequencies) == 0:
        print("Warning (SA-FM): Welch spectrum analysis yielded no frequencies. Skipping effect.")
        return song

    max_freq_index = np.argmax(spectrum)
    max_power_freq = frequencies[max_freq_index]

    mod_freq_est = max_power_freq / 2.0
    if mod_freq_est <= 1.0: mod_freq_est = 20.0 # Ensure audible modulator freq

    analytic_signal_env = signal.hilbert(song)
    amplitude_envelope = np.abs(analytic_signal_env)
    mod_amp_est = np.mean(amplitude_envelope) * 0.5 # Use mean for stability
    if mod_amp_est == 0: mod_amp_est = 0.1 # Ensure some modulation if input is not silent

    t_vector = np.arange(num_samples) / FS
    modulating_wave = mod_amp_est * np.sin(TAU * mod_freq_est * t_vector)
    
    # Simple FM: input(t) + A * sin(2*pi*fm*t)
    # For more pronounced FM: input(t) * (1 + index * sin(2*pi*fm*t)) or phase modulation
    # The original was additive, let's keep that.
    fm_wave = song + modulating_wave # Modulator has amplitude baked in

    cheby_order = 3
    # Adaptive cutoff for smoothing, related to estimated max_power_freq
    cheby_cutoff_hz = max_power_freq * 2.0 if max_power_freq > 0 else 3000.0
    cheby_cutoff_hz = np.clip(cheby_cutoff_hz, 100.0, FS / 2.1) # Ensure valid range

    Wn_cheby = cheby_cutoff_hz / (FS / 2.0)
    try:
        b, a = signal.cheby1(cheby_order, rp=0.5, Wn=Wn_cheby, btype='low') # 0.5 dB ripple
        smoothed_fm_wave = signal.filtfilt(b, a, fm_wave)
    except ValueError as e:
        print(f"Error creating Chebyshev filter for SA-FM ({e}). Skipping smoothing.")
        return fm_wave
    return smoothed_fm_wave

def apply_experimental_phase_vocoder_modulation(audio_data: np.ndarray, sample_rate: int,
                                                n_fft: int = 2048, hop_length: int = None,
                                                stretch_factor: float = 1.0, pitch_shift_semitones: float = 0.0
                                                ) -> np.ndarray:
    print(f"Applying EXPERIMENTAL Phase Vocoder Effects: StretchFactor={stretch_factor}, PitchShift={pitch_shift_semitones} semitones")
    if audio_data.size == 0: return np.copy(audio_data)
    song_float = audio_data.astype(np.float32)
    
    if hop_length is None:
        hop_length = n_fft // 4

    processed_song = song_float
    
    if pitch_shift_semitones != 0.0:
        try:
            processed_song = librosa.effects.pitch_shift(y=processed_song, sr=sample_rate,
                                                         n_steps=pitch_shift_semitones,
                                                         bins_per_octave=12, # Standard
                                                         res_type='soxr_hq', # High quality resampler
                                                         n_fft=n_fft, hop_length=hop_length)
        except Exception as e:
            print(f"Error during pitch shifting: {e}. Original audio for this step passed through.")
    
    if stretch_factor != 1.0:
        if stretch_factor <= 0:
            print("Warning (PhaseVocoderMod): Stretch factor must be positive. Skipping time stretch.")
        else:
            # librosa.effects.time_stretch expects rate: > 1.0 speeds up, < 1.0 slows down
            # If user provides stretch_factor where > 1.0 is slower, then rate = 1.0 / stretch_factor
            # Assuming stretch_factor > 1.0 means slow down (increase duration)
            time_stretch_rate = 1.0 / stretch_factor 
            try:
                processed_song = librosa.effects.time_stretch(y=processed_song, rate=time_stretch_rate,
                                                              n_fft=n_fft, hop_length=hop_length)
            except Exception as e:
                 print(f"Error during time stretching: {e}. Original audio for this step passed through.")
    return processed_song


def apply_experimental_fft_filter(audio_data: np.ndarray, sample_rate: int,
                                  threshold_factor: float = 0.01) -> np.ndarray:
    print(f"Applying EXPERIMENTAL FFT Filter (Threshold Factor={threshold_factor})")
    if audio_data.size == 0: return np.copy(audio_data)

    if not (0 < threshold_factor <= 1.0): # Allow up to 1.0 (though that would zero almost everything)
        print("Warning (FFT Filter): threshold_factor should be (0, 1]. Using 0.01.")
        threshold_factor = 0.01

    yf = np.fft.rfft(audio_data)
    yf_abs = np.abs(yf)
    if yf_abs.size == 0: return np.copy(audio_data) # Should not happen

    max_coeff_mag = np.max(yf_abs)
    if max_coeff_mag == 0: # Silent or DC only
        return np.copy(audio_data)

    dynamic_threshold = max_coeff_mag * threshold_factor
    mask = yf_abs > dynamic_threshold
    yf_clean = yf * mask
    
    return np.fft.irfft(yf_clean, n=len(audio_data))
