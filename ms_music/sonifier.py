# ms_music/sonifier.py

import numpy as np
from tqdm import tqdm
import os

from . import data_loader
from . import audio_generator
from . import effects as audio_effects
from . import utils

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
            # Optionally, re-raise or handle more gracefully depending on desired package behavior
            # For now, print error and sonifier.current_audio_data might remain the pre-effect audio.


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
        