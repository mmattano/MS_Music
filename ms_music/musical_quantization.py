# ms_music/musical_quantization.py

import numpy as np
import math

class MusicalNoteQuantizer:
    """
    A class to handle quantization of frequencies to musical notes.
    Supports different scales, tuning systems, and note selection modes.
    """
    
    # Note names for reference
    CHROMATIC_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Scales (semitone intervals from root)
    SCALES = {
        'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # All 12 semitones
        'major': [0, 2, 4, 5, 7, 9, 11],  # Major scale (Ionian)
        'minor': [0, 2, 3, 5, 7, 8, 10],  # Natural minor scale (Aeolian)
        'pentatonic_major': [0, 2, 4, 7, 9],  # Major pentatonic
        'pentatonic_minor': [0, 3, 5, 7, 10],  # Minor pentatonic
        'blues': [0, 3, 5, 6, 7, 10],  # Blues scale
        'dorian': [0, 2, 3, 5, 7, 9, 10],  # Dorian mode
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],  # Mixolydian mode
        'whole_tone': [0, 2, 4, 6, 8, 10],  # Whole tone scale
    }
    
    def __init__(self, 
                 scale='chromatic', 
                 root_note='C', 
                 tuning_freq=440.0,  # A4 frequency
                 freq_range=(80, 4000),
                 octave_range=None):
        """
        Initialize the note quantizer.
        
        Args:
            scale: Scale to use ('chromatic', 'major', 'minor', etc.)
            root_note: Root note of the scale ('C', 'D', 'E', etc.)
            tuning_freq: Frequency of A4 in Hz (standard is 440.0)
            freq_range: (min_freq, max_freq) range to generate notes for
            octave_range: (min_octave, max_octave) or None for auto-calculation
        """
        self.scale = scale
        self.root_note = root_note
        self.tuning_freq = tuning_freq
        self.freq_range = freq_range
        
        # Calculate the frequency of C4 based on A4 tuning
        # A4 is 9 semitones above C4
        self.c4_freq = tuning_freq / (2 ** (9/12))
        
        # Generate all available note frequencies
        self.note_frequencies = self._generate_note_frequencies(octave_range)
        
        print(f"Initialized MusicalNoteQuantizer:")
        print(f"  Scale: {scale} in {root_note}")
        print(f"  Tuning: A4 = {tuning_freq} Hz (C4 = {self.c4_freq:.2f} Hz)")
        print(f"  Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
        print(f"  Available notes: {len(self.note_frequencies)}")
    
    def _generate_note_frequencies(self, octave_range=None):
        """Generate all note frequencies within the specified range."""
        if octave_range is None:
            # Auto-calculate octave range based on frequency range
            min_octave = max(0, int(math.log2(self.freq_range[0] / self.c4_freq)) + 4 - 1)
            max_octave = min(10, int(math.log2(self.freq_range[1] / self.c4_freq)) + 4 + 1)
        else:
            min_octave, max_octave = octave_range
        
        note_frequencies = []
        scale_intervals = self.SCALES[self.scale]
        root_semitone = self.CHROMATIC_NOTES.index(self.root_note)
        
        for octave in range(min_octave, max_octave + 1):
            for interval in scale_intervals:
                # Calculate semitone position relative to C4
                semitone_from_c4 = (octave - 4) * 12 + root_semitone + interval
                
                # Calculate frequency
                frequency = self.c4_freq * (2 ** (semitone_from_c4 / 12))
                
                # Only include if within frequency range
                if self.freq_range[0] <= frequency <= self.freq_range[1]:
                    note_name = self.CHROMATIC_NOTES[(root_semitone + interval) % 12]
                    note_frequencies.append({
                        'frequency': frequency,
                        'note': note_name,
                        'octave': octave,
                        'semitone_from_c4': semitone_from_c4
                    })
        
        # Sort by frequency
        note_frequencies.sort(key=lambda x: x['frequency'])
        return note_frequencies
    
    def quantize_frequency(self, frequency):
        """
        Find the closest musical note frequency to the given frequency.
        
        Args:
            frequency: Input frequency in Hz
            
        Returns:
            dict: Information about the closest note
        """
        if not self.note_frequencies:
            return {'frequency': frequency, 'note': 'N/A', 'octave': 0, 'distance': float('inf')}
        
        # Find closest note by minimum frequency distance
        min_distance = float('inf')
        closest_note = None
        
        for note_info in self.note_frequencies:
            distance = abs(note_info['frequency'] - frequency)
            if distance < min_distance:
                min_distance = distance
                closest_note = note_info.copy()
                closest_note['original_frequency'] = frequency
                closest_note['distance'] = distance
        
        return closest_note
    
    def quantize_frequency_log(self, frequency):
        """
        Find the closest musical note using logarithmic (musical) distance.
        This considers the ratio of frequencies rather than absolute difference.
        """
        if not self.note_frequencies:
            return {'frequency': frequency, 'note': 'N/A', 'octave': 0, 'distance': float('inf')}
        
        if frequency <= 0:
            return self.note_frequencies[0]
        
        # Find closest note by minimum logarithmic distance
        min_distance = float('inf')
        closest_note = None
        
        log_freq = math.log(frequency)
        
        for note_info in self.note_frequencies:
            log_note_freq = math.log(note_info['frequency'])
            distance = abs(log_note_freq - log_freq)
            if distance < min_distance:
                min_distance = distance
                closest_note = note_info.copy()
                closest_note['original_frequency'] = frequency
                closest_note['log_distance'] = distance
        
        return closest_note
    
    def get_note_frequencies_list(self):
        """Return a list of all available note frequencies."""
        return [note['frequency'] for note in self.note_frequencies]
    
    def print_available_notes(self):
        """Print all available notes for debugging/reference."""
        print("Available notes:")
        for note in self.note_frequencies:
            print(f"  {note['note']}{note['octave']}: {note['frequency']:.2f} Hz")


def mz_to_frequency_inverse_log(mz_value, mz_min, mz_max, 
                               freq_min=200.0, freq_max=4000.0):
    """Maps m/z values to frequencies using inverse logarithmic scaling."""
    if mz_value <= 0 or mz_min <= 0:
        return freq_min
    
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    
    log_ratio = math.log(freq_max / freq_min)
    frequency = freq_max * math.exp(-mz_normalized * log_ratio)
    
    return frequency


def mz_to_frequency_power_law(mz_value, mz_min, mz_max,
                             freq_min=200.0, freq_max=4000.0, 
                             exponent=1.5):
    """Maps m/z values using a power law relationship."""
    if mz_value <= 0:
        return freq_min
        
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    
    freq_normalized = (1 - mz_normalized) ** exponent
    
    log_freq_min = math.log(freq_min)
    log_freq_max = math.log(freq_max)
    log_frequency = log_freq_min + freq_normalized * (log_freq_max - log_freq_min)
    
    return math.exp(log_frequency)


def mz_to_frequency_musical_octaves(mz_value, mz_min, mz_max,
                                   base_freq=440.0, num_octaves=4):
    """Maps m/z to frequencies using musical octaves."""
    if mz_value <= 0:
        return base_freq
        
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    
    mz_inverted = 1 - mz_normalized
    octave_position = mz_inverted * num_octaves
    frequency = base_freq * (2 ** octave_position)
    
    return frequency


def mz_to_frequency_chromatic_scale(mz_value, mz_min, mz_max,
                                   base_freq=261.63, 
                                   num_semitones=48):
    """Maps m/z to frequencies using chromatic scale."""
    if mz_value <= 0:
        return base_freq
        
    mz_normalized = (mz_value - mz_min) / (mz_max - mz_min)
    mz_normalized = np.clip(mz_normalized, 0, 1)
    mz_inverted = 1 - mz_normalized
    
    semitone_position = mz_inverted * num_semitones
    frequency = base_freq * (2 ** (semitone_position / 12))
    
    return frequency