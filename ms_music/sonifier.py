# ms_music/sonifier.py

import numpy as np
from tqdm import tqdm
import os
import math

from . import data_loader
from . import audio_generator
from . import effects as audio_effects
from . import utils
from . import musical_quantization
from .musical_quantization import MusicalNoteQuantizer

class MSSonifier:
    """
    Main class for orchestrating the sonification of mass spectrometry data.
    """
    def __init__(self, filepath: str, ms_level: int = 1, total_duration_minutes: float = 10, sample_rate: int = 44100):
        self.filepath = filepath
        self.ms_level = ms_level
        self.total_duration_seconds = total_duration_minutes * 60
        self.sample_rate = sample_rate
        self.metadata_harmonization = False

        self.raw_spectra = None
        self.processed_spectra_dfs = None
        self.max_intensity_overall = 0
        self.min_mz_overall = 0
        self.max_mz_overall = 0
        
        self.current_audio_data = None
        self.note_quantizer = None

        print(f"MSSonifier initialized for file: {os.path.basename(filepath)}")
        print(f"Target audio duration: {total_duration_minutes} minutes, Sample rate: {sample_rate} Hz")

    def load_and_preprocess_data(self, scan_ratio_range=None):
        print("Loading mzML data...")
        self.raw_spectra = data_loader.load_mzml_data(
            self.filepath, 
            ms_level=self.ms_level, 
            metadata_harmonization=self.metadata_harmonization
        )
        
        if not self.raw_spectra:
            print("No spectra found during loading. Aborting.")
            return

        if scan_ratio_range:
            if not (isinstance(scan_ratio_range, tuple) and len(scan_ratio_range) == 2 and
                    0.0 <= scan_ratio_range[0] < scan_ratio_range[1] <= 1.0):
                # Default to full range if invalid, or raise error. Let's default with a warning.
                print(f"Warning: Invalid scan_ratio_range {scan_ratio_range}. Processing all scans.")
                scan_ratio_range = None # Process all
            
            if scan_ratio_range: # Re-check after potential reset
                num_spectra = len(self.raw_spectra)
                start_index = int(num_spectra * scan_ratio_range[0])
                end_index = int(num_spectra * scan_ratio_range[1])
                self.raw_spectra = self.raw_spectra[start_index:end_index]
                print(f"Selected {len(self.raw_spectra)} scans based on scan_ratio_range {scan_ratio_range}.")

        if not self.raw_spectra: # Check if slicing resulted in empty list
            print("No spectra selected after applying scan_ratio_range. Aborting.")
            return

        print(f"Loaded {len(self.raw_spectra)} spectra for preprocessing.")
        
        print("Preprocessing spectra...")
        (self.processed_spectra_dfs, 
         self.max_intensity_overall, 
         self.min_mz_overall, 
         self.max_mz_overall) = data_loader.preprocess_spectra(self.raw_spectra)
        
        if not self.processed_spectra_dfs:
            print("Preprocessing failed or resulted in no processable spectra. Aborting.")
            return

        print(f"Preprocessing complete. Found {len(self.processed_spectra_dfs)} processable spectra objects.")
        print(f"Overall Max Intensity: {self.max_intensity_overall:.2e}")
        print(f"Overall m/z Range: {self.min_mz_overall:.2f} - {self.max_mz_overall:.2f}")

    def sonify(self, method: str = 'gradient', method_params: dict = None):
        if self.processed_spectra_dfs is None or not self.processed_spectra_dfs:
            print("Data not loaded or preprocessed. Please call load_and_preprocess_data() first.")
            return

        if method_params is None:
            method_params = {}
            
        print(f"Starting sonification using '{method}' method...")

        if method == 'gradient':
            overlap_percentage = method_params.get('overlap_percentage', 0.05)
            self.current_audio_data = audio_generator.generate_audio_gradient_method(
                processed_spectra_dfs=self.processed_spectra_dfs,
                max_intensity_overall=self.max_intensity_overall,
                min_mz_overall=self.min_mz_overall,
                max_mz_overall=self.max_mz_overall,
                total_duration_seconds=self.total_duration_seconds,
                sample_rate=self.sample_rate,
                overlap_percentage=overlap_percentage
            )
        elif method == 'adsr':
            adsr_settings = method_params.get('adsr_settings', None) # Pass None to use defaults in generator
            self.current_audio_data = audio_generator.generate_audio_adsr_method(
                processed_spectra_dfs=self.processed_spectra_dfs,
                max_intensity_overall=self.max_intensity_overall,
                total_duration_seconds=self.total_duration_seconds,
                sample_rate=self.sample_rate,
                adsr_settings=adsr_settings
            )
        else:
            # Raising an error is more decisive than printing a warning and continuing
            raise ValueError(f"Unknown sonification method: {method}. Choose 'gradient' or 'adsr'.")

        if self.current_audio_data is not None and self.current_audio_data.size > 0:
            print("Sonification complete.")
        else:
            print("Sonification failed to produce audio data or produced empty audio.")
            self.current_audio_data = None # Ensure it's None if generation failed

    def apply_effect(self, effect_name: str, effect_params: dict = None):
        if self.current_audio_data is None or self.current_audio_data.size == 0:
            print("No audio data to apply effects to. Please sonify first.")
            return

        if effect_params is None:
            effect_params = {}

        try:
            # Construct function name string and get the function object
            effect_function_name = f"apply_{effect_name}" # Standardized naming convention
            effect_function = getattr(audio_effects, effect_function_name)
            
            print(f"Applying effect: {effect_name} with params: {effect_params}")
            
            # Pass sample_rate implicitly if the function expects it (most do)
            # The effect functions themselves should handle their specific parameters.
            # We pass self.sample_rate and let **effect_params unpack the rest.
            self.current_audio_data = effect_function(
                self.current_audio_data, 
                self.sample_rate, 
                **effect_params
            )
            print(f"Effect '{effect_name}' applied.")
        except AttributeError:
            print(f"Error: Effect function '{effect_function_name}' not found in effects module.")
        except Exception as e:
            print(f"Error applying effect '{effect_name}': {e}")
            self.current_audio_data = None

    def save_audio(self, output_filepath: str, normalize: bool = True):
        if self.current_audio_data is None or self.current_audio_data.size == 0:
            print("No audio data to save. Please sonify first.")
            return

        audio_to_save = self.current_audio_data
        if normalize:
            print("Normalizing audio for saving...")
            audio_to_save = utils.normalize_audio_to_16bit(audio_to_save)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
            
        print(f"Saving audio to {output_filepath}...")
        utils.save_wav(output_filepath, audio_to_save, self.sample_rate)
        # save_wav now has its own print message

    def get_current_audio(self, copy=True): # Added copy parameter
        """
        Returns the current audio data.

        Args:
            copy (bool, optional): If True (default), returns a copy of the audio data
                                   to prevent external modifications. If False, returns
                                   a direct reference (use with caution).
        Returns:
            np.ndarray or None: The current audio data, or None if not generated.
        """
        if self.current_audio_data is None:
            return None
        return np.copy(self.current_audio_data) if copy else self.current_audio_data
    

    def sonify_enhanced(self, method: str = 'gradient_enhanced', method_params: dict = None):
        """
        Enhanced sonification with better frequency mapping options.
        
        Args:
            method: 'gradient_enhanced' or 'adsr_enhanced'
            method_params: Dictionary with parameters including:
                - frequency_mapping: 'inverse_log', 'power_law', 'musical_octaves', 'chromatic'
                - freq_range: (min_freq, max_freq) tuple in Hz
                - overlap_percentage: for gradient method
                - adsr_settings: for ADSR method
        """
        if self.processed_spectra_dfs is None or not self.processed_spectra_dfs:
            print("Data not loaded or preprocessed. Please call load_and_preprocess_data() first.")
            return

        if method_params is None:
            method_params = {}
            
        # Default parameters
        frequency_mapping = method_params.get('frequency_mapping', 'inverse_log')
        freq_range = method_params.get('freq_range', (200.0, 4000.0))
        
        print(f"Starting enhanced sonification using '{method}' method...")
        print(f"Frequency mapping: {frequency_mapping}, Range: {freq_range[0]}-{freq_range[1]} Hz")

        if method == 'gradient_enhanced':
            overlap_percentage = method_params.get('overlap_percentage', 0.05)
            self.current_audio_data = self._generate_audio_gradient_enhanced(
                frequency_mapping=frequency_mapping,
                freq_range=freq_range,
                overlap_percentage=overlap_percentage
            )
        elif method == 'adsr_enhanced':
            adsr_settings = method_params.get('adsr_settings', None)
            self.current_audio_data = self._generate_audio_adsr_enhanced(
                frequency_mapping=frequency_mapping,
                freq_range=freq_range,
                adsr_settings=adsr_settings
            )
        else:
            raise ValueError(f"Unknown enhanced method: {method}. Choose 'gradient_enhanced' or 'adsr_enhanced'.")

        if self.current_audio_data is not None and self.current_audio_data.size > 0:
            print("Enhanced sonification complete.")
        else:
            print("Enhanced sonification failed to produce audio data.")
            self.current_audio_data = None

    def _mz_to_frequency(self, mz_value, mapping_type, freq_range):
        """Helper method for frequency mapping."""
        freq_min, freq_max = freq_range
        
        if mapping_type == 'inverse_log':
            if mz_value <= 0 or self.min_mz_overall <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (self.max_mz_overall - self.min_mz_overall)
            mz_normalized = np.clip(mz_normalized, 0, 1)
            log_ratio = math.log(freq_max / freq_min)
            return freq_max * math.exp(-mz_normalized * log_ratio)
            
        elif mapping_type == 'power_law':
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (self.max_mz_overall - self.min_mz_overall)
            mz_normalized = np.clip(mz_normalized, 0, 1)
            freq_normalized = (1 - mz_normalized) ** 1.5  # Power law exponent
            log_freq_min = math.log(freq_min)
            log_freq_max = math.log(freq_max)
            log_frequency = log_freq_min + freq_normalized * (log_freq_max - log_freq_min)
            return math.exp(log_frequency)
            
        elif mapping_type == 'musical_octaves':
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (self.max_mz_overall - self.min_mz_overall)
            mz_normalized = np.clip(mz_normalized, 0, 1)
            mz_inverted = 1 - mz_normalized
            num_octaves = math.log2(freq_max / freq_min)  # Calculate octaves from freq range
            octave_position = mz_inverted * num_octaves
            return freq_min * (2 ** octave_position)
            
        elif mapping_type == 'chromatic':
            if mz_value <= 0:
                return freq_min
            mz_normalized = (mz_value - self.min_mz_overall) / (self.max_mz_overall - self.min_mz_overall)
            mz_normalized = np.clip(mz_normalized, 0, 1)
            mz_inverted = 1 - mz_normalized
            num_semitones = 12 * math.log2(freq_max / freq_min)  # Semitones in the range
            semitone_position = mz_inverted * num_semitones
            return freq_min * (2 ** (semitone_position / 12))
            
        else:  # Fallback to original linear
            return float(mz_value)

    def _generate_audio_gradient_enhanced(self, frequency_mapping, freq_range, overlap_percentage):
        """Enhanced gradient method with better frequency mapping."""
        
        num_scans = len(self.processed_spectra_dfs)
        samples_per_scan = round(self.sample_rate * self.total_duration_seconds / num_scans)
        total_samples = int(samples_per_scan * num_scans)
        
        time_vector = np.linspace(0, total_samples / self.sample_rate, total_samples, 
                                endpoint=False, dtype=np.float32)
        
        # Get all unique m/z values
        all_mz_values = set()
        for scan_df in self.processed_spectra_dfs:
            if not scan_df.empty:
                all_mz_values.update(scan_df.index)
        
        song = np.zeros(total_samples, dtype=np.float32)
        overlap_samples = round(overlap_percentage * samples_per_scan)
        
        print("Generating enhanced audio with improved frequency mapping...")
        
        for mz_value in all_mz_values:
            if mz_value <= 0:
                continue
                
            # Convert m/z to frequency using selected mapping
            frequency = self._mz_to_frequency(mz_value, frequency_mapping, freq_range)
            if frequency <= 0:
                continue
                
            # Generate sine wave at this frequency
            mz_sine_wave = np.sin(frequency * 2 * math.pi * time_vector)
            modulated_mz_wave_component = np.zeros_like(mz_sine_wave)

            # Get normalized intensities for this m/z across all scans
            normalized_intensities_for_mz = np.zeros(num_scans, dtype=np.float32)
            for i, scan_df in enumerate(self.processed_spectra_dfs):
                if mz_value in scan_df.index:
                    normalized_intensities_for_mz[i] = scan_df.loc[mz_value, 'intensities'] / self.max_intensity_overall
            
            # Apply intensity modulation (simplified version - you can use the full ramp version)
            for scan_idx in range(num_scans):
                start_sample_idx = scan_idx * samples_per_scan
                end_sample_idx = start_sample_idx + samples_per_scan
                
                current_segment_sine = mz_sine_wave[start_sample_idx:end_sample_idx]
                if current_segment_sine.size == 0:
                    continue

                current_intensity = normalized_intensities_for_mz[scan_idx]
                modulated_mz_wave_component[start_sample_idx:end_sample_idx] = current_segment_sine * current_intensity
            
            song += modulated_mz_wave_component

        return song
    
    def setup_musical_quantization(self, 
                                  scale='chromatic', 
                                  root_note='C', 
                                  tuning_freq=440.0,
                                  freq_range=None):
        """
        Set up musical note quantization for sonification.
        
        Args:
            scale: Musical scale ('chromatic', 'major', 'minor', 'pentatonic_major', 
                   'pentatonic_minor', 'blues', 'dorian', 'mixolydian', 'whole_tone')
            root_note: Root note of the scale ('C', 'D', 'E', 'F', 'G', 'A', 'B')
            tuning_freq: Frequency of A4 in Hz (standard is 440.0)
            freq_range: (min_freq, max_freq) or None to use default (200, 4000)
        """
        if freq_range is None:
            freq_range = (200, 4000)
            
        self.note_quantizer = MusicalNoteQuantizer(
            scale=scale,
            root_note=root_note,
            tuning_freq=tuning_freq,
            freq_range=freq_range
        )
        
        print(f"Musical quantization set up: {scale} scale in {root_note}")
        return self.note_quantizer
    
    def setup_musical_quantization(self, 
                                scale='chromatic', 
                                root_note='C', 
                                tuning_freq=440.0,
                                freq_range=None):
        """
        Set up musical note quantization for sonification.
        
        Args:
            scale: Musical scale ('chromatic', 'major', 'minor', 'pentatonic_major', 
                'pentatonic_minor', 'blues', 'dorian', 'mixolydian', 'whole_tone')
            root_note: Root note of the scale ('C', 'D', 'E', 'F', 'G', 'A', 'B')
            tuning_freq: Frequency of A4 in Hz (standard is 440.0)
            freq_range: (min_freq, max_freq) or None to use default (200, 4000)
        """
        if freq_range is None:
            freq_range = (200, 4000)
            
        self.note_quantizer = MusicalNoteQuantizer(
            scale=scale,
            root_note=root_note,
            tuning_freq=tuning_freq,
            freq_range=freq_range
        )
        
        print(f"Musical quantization set up: {scale} scale in {root_note}")
        return self.note_quantizer
    
    def sonify_quantized(self, 
                        base_mapping='inverse_log',
                        method_params=None):
        """
        Sonify using quantized musical notes.
        
        Args:
            base_mapping: Base frequency mapping before quantization
                        ('inverse_log', 'power_law', 'musical_octaves', 'chromatic')
            method_params: Dictionary with parameters:
                - freq_range: (min_freq, max_freq) for base mapping
                - overlap_percentage: for intensity transitions
                - use_log_distance: Use logarithmic distance for note selection
                - scale: Musical scale to use
                - root_note: Root note of the scale
                - tuning_freq: A4 frequency
        """
        if self.processed_spectra_dfs is None or not self.processed_spectra_dfs:
            print("Data not loaded or preprocessed. Please call load_and_preprocess_data() first.")
            return

        if method_params is None:
            method_params = {}
        
        # Set up quantizer if not already done or if parameters changed
        scale = method_params.get('scale', 'major')
        root_note = method_params.get('root_note', 'C')
        tuning_freq = method_params.get('tuning_freq', 440.0)
        freq_range = method_params.get('freq_range', (200, 3000))
        
        if (self.note_quantizer is None or 
            self.note_quantizer.scale != scale or 
            self.note_quantizer.root_note != root_note or
            self.note_quantizer.tuning_freq != tuning_freq):
            
            self.setup_musical_quantization(scale, root_note, tuning_freq, freq_range)
        
        overlap_percentage = method_params.get('overlap_percentage', 0.05)
        use_log_distance = method_params.get('use_log_distance', True)
        
        print(f"Starting quantized sonification with {base_mapping} base mapping...")
        print(f"Scale: {scale} in {root_note}, Tuning: A4 = {tuning_freq} Hz")
        
        self.current_audio_data = self._generate_quantized_audio(
            base_mapping=base_mapping,
            freq_range=freq_range,
            overlap_percentage=overlap_percentage,
            use_log_distance=use_log_distance
        )
        
        if self.current_audio_data is not None and self.current_audio_data.size > 0:
            print("Quantized sonification complete.")
        else:
            print("Quantized sonification failed to produce audio data.")
            self.current_audio_data = None
    
    def _generate_quantized_audio(self, base_mapping, freq_range, overlap_percentage, use_log_distance):
        """Generate audio with quantized frequencies."""
        from tqdm import tqdm
        
        num_scans = len(self.processed_spectra_dfs)
        samples_per_scan = round(self.sample_rate * self.total_duration_seconds / num_scans)
        total_samples = int(samples_per_scan * num_scans)
        
        time_vector = np.linspace(0, total_samples / self.sample_rate, total_samples, 
                                endpoint=False, dtype=np.float32)
        
        # Get all unique m/z values
        all_mz_values = set()
        for scan_df in self.processed_spectra_dfs:
            if not scan_df.empty:
                all_mz_values.update(scan_df.index)
        
        if not all_mz_values:
            return np.zeros(total_samples, dtype=np.float32)
        
        song = np.zeros(total_samples, dtype=np.float32)
        overlap_samples = round(overlap_percentage * samples_per_scan)
        
        # Create mapping from m/z to quantized frequencies
        mz_to_freq_map = {}
        unique_frequencies = set()
        
        print("Mapping m/z values to quantized frequencies...")
        for mz_value in tqdm(sorted(all_mz_values), desc="Quantizing frequencies"):
            if mz_value <= 0:
                continue
                
            # Get continuous frequency using base mapping
            if base_mapping == 'inverse_log':
                continuous_freq = self._mz_to_frequency_inverse_log(mz_value, freq_range)
            elif base_mapping == 'power_law':
                continuous_freq = self._mz_to_frequency_power_law(mz_value, freq_range)
            elif base_mapping == 'musical_octaves':
                continuous_freq = self._mz_to_frequency_musical_octaves(mz_value, freq_range)
            elif base_mapping == 'chromatic':
                continuous_freq = self._mz_to_frequency_chromatic(mz_value, freq_range)
            else:
                continuous_freq = float(mz_value)
            
            # Quantize to nearest musical note
            if use_log_distance:
                note_info = self.note_quantizer.quantize_frequency_log(continuous_freq)
            else:
                note_info = self.note_quantizer.quantize_frequency(continuous_freq)
            
            quantized_freq = note_info['frequency']
            mz_to_freq_map[mz_value] = quantized_freq
            unique_frequencies.add(quantized_freq)
        
        print(f"Mapped {len(all_mz_values)} m/z values to {len(unique_frequencies)} unique frequencies")
        
        # Group m/z values by their quantized frequency for more efficient processing
        freq_to_mz_groups = {}
        for mz_value, freq in mz_to_freq_map.items():
            if freq not in freq_to_mz_groups:
                freq_to_mz_groups[freq] = []
            freq_to_mz_groups[freq].append(mz_value)
        
        print("Generating audio with quantized frequencies...")
        for frequency, mz_group in tqdm(freq_to_mz_groups.items(), desc="Processing frequencies"):
            if frequency <= 0:
                continue
            
            # Generate sine wave at this quantized frequency
            mz_sine_wave = np.sin(frequency * 2 * math.pi * time_vector)
            modulated_mz_wave_component = np.zeros_like(mz_sine_wave)
            
            # Combine intensities from all m/z values that map to this frequency
            combined_intensities = np.zeros(num_scans, dtype=np.float32)
            for mz_value in mz_group:
                for i, scan_df in enumerate(self.processed_spectra_dfs):
                    if mz_value in scan_df.index:
                        combined_intensities[i] += scan_df.loc[mz_value, 'intensities'] / self.max_intensity_overall
            
            # Apply intensity modulation with simple transitions
            for scan_idx in range(num_scans):
                start_sample_idx = scan_idx * samples_per_scan
                end_sample_idx = start_sample_idx + samples_per_scan
                
                current_segment_sine = mz_sine_wave[start_sample_idx:end_sample_idx]
                if current_segment_sine.size == 0:
                    continue

                current_intensity = combined_intensities[scan_idx]
                
                # Simple intensity modulation
                modulated_mz_wave_component[start_sample_idx:end_sample_idx] = current_segment_sine * current_intensity
            
            song += modulated_mz_wave_component
        
        return song
    
    def _mz_to_frequency_inverse_log(self, mz_value, freq_range):
        """Helper method for inverse log mapping."""
        from .musical_quantization import mz_to_frequency_inverse_log
        return mz_to_frequency_inverse_log(mz_value, self.min_mz_overall, self.max_mz_overall, *freq_range)
    
    def _mz_to_frequency_power_law(self, mz_value, freq_range, exponent=1.5):
        """Helper method for power law mapping."""
        from .musical_quantization import mz_to_frequency_power_law
        return mz_to_frequency_power_law(mz_value, self.min_mz_overall, self.max_mz_overall, *freq_range, exponent)
    
    def _mz_to_frequency_musical_octaves(self, mz_value, freq_range):
        """Helper method for musical octaves mapping."""
        from .musical_quantization import mz_to_frequency_musical_octaves
        return mz_to_frequency_musical_octaves(mz_value, self.min_mz_overall, self.max_mz_overall, freq_range[0], 4)
    
    def _mz_to_frequency_chromatic(self, mz_value, freq_range):
        """Helper method for chromatic mapping."""
        from .musical_quantization import mz_to_frequency_chromatic_scale
        return mz_to_frequency_chromatic_scale(mz_value, self.min_mz_overall, self.max_mz_overall, freq_range[0], 48)
    