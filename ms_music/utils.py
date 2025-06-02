# ms_music/utils.py

import numpy as np
import wavio
import os

def normalize_audio_to_16bit(audio_data: np.ndarray):
    """
    Normalizes floating point audio data to the 16-bit integer range.
    """
    if audio_data is None or audio_data.size == 0:
        return np.array([], dtype=np.int16)

    max_abs_val = np.max(np.abs(audio_data))
    if max_abs_val < 1e-9: 
        return np.zeros_like(audio_data, dtype=np.int16)
        
    normalized_float = audio_data / max_abs_val 
    scaled_audio = normalized_float * (2**15 - 1)
    return scaled_audio.astype(np.int16)

def save_wav(filepath: str, audio_data: np.ndarray, sample_rate: int, sampwidth: int = 2):
    """
    Saves audio data to a .wav file using wavio.
    Assumes audio_data is int16 if sampwidth is 2.
    """
    if audio_data is None or audio_data.size == 0:
        print(f"Error (Save WAV): No audio data provided to save to {filepath}.")
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory for WAV output: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}. Cannot save WAV file.")
            return

    try:
        if sampwidth == 2 and audio_data.dtype != np.int16:
            print(f"Warning (Save WAV): Audio data dtype is {audio_data.dtype}, not int16 for sampwidth=2. "
                  "Ensure audio was normalized to 16-bit range if it was float.")

            if np.issubdtype(audio_data.dtype, np.floating):
                 print("Attempting to convert float audio to int16 for saving. Max value before conversion:", np.max(np.abs(audio_data)))
                 audio_to_save = (audio_data / np.max(np.abs(audio_data)) * (2**15 - 1)).astype(np.int16) if np.max(np.abs(audio_data)) > 0 else np.zeros_like(audio_data, dtype=np.int16)
            else:
                 audio_to_save = audio_data.astype(np.int16) 
        else:
            audio_to_save = audio_data
        
        wavio.write(filepath, audio_to_save, sample_rate, sampwidth=sampwidth)
        print(f"Audio successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving .wav file to {filepath}: {e}")