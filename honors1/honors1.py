from __future__ import annotations
import numpy as np
from scipy.io.wavfile import write
from typing import Optional
import os

# Constants
AMPLITUDE = 4096
SAMPLERATE = 44100
PITCH_CLASSES = {
    'C': 0, 'c': 1, 'D': 2, 'd': 3,
    'E': 4, 'F': 5, 'f': 6, 'G': 7, 'g': 8,
    'A': 9, 'a': 10, 'B': 11
}
DURATION_MAP = {
    "WN": 4.0, "DHN": 3.0, "DDHN": 3.5, "HN": 2.0, "HNT": 4.0 / 3,
    "QN": 1.0, "QNT": 2.0 / 3, "DQN": 1.5, "DDQN": 1.75, "EN": 0.5,
    "DEN": 0.75, "ENT": 1.0 / 3, "DDEN": 0.875, "SN": 0.25, "DSN": 0.375,
    "SNT": 1.0 / 6, "TN": 0.125, "TNT": 1.0 / 12
}
TEMPOS = {
    "Grave": 30, "Lento": 40, "Largo": 50, "Adagio": 60, "Adagietto": 70,
    "Andante": 75, "Moderato": 90, "Allegretto": 100, "Allegro": 120,
    "Vivace": 135, "Presto": 170, "Prestissimo": 180
}

output_directory = "output_songs"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

class Wave:
    def __init__(self, frequency: float, duration: float, amplitude: int = AMPLITUDE, data: Optional[np.ndarray] = None):
        """
        Initializes a Wave object with the specified frequency, duration, and Optional amplitude.
        Amplitude is set by default to 4096.
        
        :param frequency: The frequency of the wave in Hz.
        :param duration: The duration of the wave in seconds.
        :param amplitude: The amplitude of the wave.
        :param data: Optional precomputed numpy array representing the wave.
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.duration = duration
        self.data = self.generate_wave() if data is None else data

    def generate_wave(self) -> np.ndarray:
        """
        Generates a sine wave array using the wave's attributes.
        
        :returns: A numpy array representing the note's wave.
        """
        t = np.linspace(0, self.duration, int(SAMPLERATE * self.duration), endpoint=False)
        wave = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        return wave

    def __add__(self, other: Wave) -> Wave:
        """
        Adds one wave to another wave, combining their data arrays.
        
        :param other: Another Wave object.
        :returns: A new Wave object with combined data.
        """
        max_length = max(len(self.data), len(other.data))
        combined_data = np.zeros(max_length)
        combined_data[:len(self.data)] += self.data
        combined_data[:len(other.data)] += other.data
        return Wave(frequency=self.frequency, duration=self.duration, data=combined_data)

class Note:
    def __init__(self, pitch: str, octave: int, duration_symbol: str, tempo: str, amplitude: int = AMPLITUDE):
        """
        Initializes a Note object which combines musical properties to create sound.
        
        :param pitch: The musical pitch of the note.
        :param octave: The octave number of the note.
        :param duration_symbol: Symbol representing the duration of the note.
        :param tempo: The tempo at which the note is played.
        :param amplitude: The amplitude of the note's sound, by default 4096.
        """
        self.pitch = pitch
        self.octave = octave
        self.duration_symbol = duration_symbol
        self.tempo = TEMPOS[tempo]
        self.frequency = self.calculate_frequency() if pitch else 0
        self.duration = self.calculate_duration()
        self.amplitude = amplitude if pitch else 0
        self.wave = Wave(self.frequency, self.duration, self.amplitude)

    def calculate_frequency(self) -> float:
        """
        Calculates the frequency of the note based on its pitch and octave.
        Base note is C in fourth octave with frequency 262 Hz.
        
        :returns: The frequency of the note in Hz.
        """
        if not self.pitch:
            return 0
        base_frequency = 262
        note_step = PITCH_CLASSES.get(self.pitch, 0)
        octave_offset = self.octave - 4
        return base_frequency * (2 ** (note_step / 12)) * (2 ** octave_offset)

    def calculate_duration(self) -> float:
        """
        Calculates the duration of the note based on the tempo and duration symbol.
        
        :returns: The duration of the note in seconds.
        """
        beats_per_second = 60 / self.tempo
        duration_notation = DURATION_MAP[self.duration_symbol]
        return duration_notation * beats_per_second

class Piano:
    def __init__(self):
        """
        Initializes a Piano object to play musical notes.
        """
        self.notes = []

    def add_note(self, note: Note) -> None:
        """
        Adds a note to the piano's list of notes.
        
        :param note: The Note object to be added.
        """
        self.notes.append(note)

    def get_combined_wave(self) -> np.ndarray:
        """
        Combines all notes in the piano into a single wave array.
        
        :returns: A numpy array representing the combined wave of all notes sequentially.
        """
        total_length = sum(note.wave.data.size for note in self.notes)
        combined_wave_data = np.zeros(total_length)
        current_index = 0
        for note in self.notes:
            length = note.wave.data.size
            combined_wave_data[current_index:current_index + length] = note.wave.data
            current_index += length
        return combined_wave_data

class Song:
    def __init__(self, filepath: str):
        """
        Initializes a Song object that can load notes from a file and play them using a Piano.
        
        :param filepath: The path to the file containing song data.
        """
        self.filepath = filepath
        self.filename_output = os.path.join(output_directory, os.path.basename(filepath).replace('.txt', '.wav'))
        self.piano = Piano()

    def load_and_play_song(self) -> None:
        """
        Loads notes from a file, adds them to the piano, and writes the resulting wave to a file.
        """
        file = open(self.filepath, 'r')
        tempos = file.readline().strip().split(',')
        lines = file.readlines()
        for tempo in tempos:
            for line in lines:
                line = line.strip()
                if not line.startswith('"'):
                    for note_info in line.strip().split('-'):
                        if note_info:
                            note, duration_symbol = note_info.split(',')
                            pitch = note[0] if len(note) > 1 else None
                            octave = int(note[1]) if pitch else 0
                            note = Note(pitch, octave, duration_symbol, tempo)
                            self.piano.add_note(note)
        write(self.filename_output, SAMPLERATE, self.piano.get_combined_wave().astype(np.int16))
        print(f"Check the generated WAV file: {self.filename_output}")

def main():
    """
    Main function to load and play songs listed.
    """
    songs = ['alouette.txt', 'row_row.txt', 'twinkle_twinkle.txt']
    for song in songs:
        my_song = Song(filepath=song)
        my_song.load_and_play_song()

if __name__ == "__main__":
    main()