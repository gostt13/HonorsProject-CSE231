from __future__ import annotations
import numpy as np
from scipy.io.wavfile import write
import os
from typing import List, Optional, Tuple

# Constants
AMPLITUDE = 4096
SAMPLERATE = 44100
OVERTONE_FACTORS = [0.36046922, 0.50991279, 0.11674297, 0.01287502]
DECAY_COEFFICIENTS = [0.25656511, 1.64549261]
PITCH_CLASSES = {'C': 0, 'c': 1, 'D': 2, 'd': 3, 'E': 4, 'F': 5, 'f': 6, 'G': 7, 'g': 8, 'A': 9, 'a': 10, 'B': 11}
DURATION_MAP = {
    "WN": 4.0, "DHN": 3.0, "DDHN": 3.5, "HN": 2.0, "HNT": 4.0/3, "QN": 1.0, "QNT": 2.0/3,
    "DQN": 1.5, "DDQN": 1.75, "EN": 0.5, "DEN": 0.75, "ENT": 1.0/3, "DDEN": 0.875, "SN": 0.25,
    "DSN": 0.375, "SNT": 1.0/6, "TN": 0.125, "TNT": 1.0/12
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
    def __init__(self, frequency: float, duration: float, data: Optional[np.ndarray] = None):
        """
        Initializes a Wave object with specified frequency and duration.

        :param frequency: The frequency of the wave in Hz.
        :param duration: The duration of the wave in seconds.
        :param data: Optional precomputed numpy array representing the wave.
        """
        self.frequency = frequency
        self.duration = duration
        self.data = data if data is not None else self.generate_wave()

    def generate_wave(self) -> np.ndarray:
        """
        Generates a sine wave using specified attributes of the Wave instance.

        :return: A numpy array representing the note's wave.
        """
        t = np.linspace(0, self.duration, int(SAMPLERATE * self.duration), endpoint=False)
        envelope = np.exp(-DECAY_COEFFICIENTS[0] - DECAY_COEFFICIENTS[1] * t)
        wave = np.zeros_like(t)
        for i in range(4):
            wave += AMPLITUDE * envelope * OVERTONE_FACTORS[i] * np.sin(2 * np.pi * i * self.frequency * t)
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
        return Wave(frequency=self.frequency + other.frequency, duration=self.duration, data=combined_data)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Wave object.
        
        :returns: A string representation of the Wave object.
        """
        return f"Wave(frequency={self.frequency}, duration={self.duration}, length of the wave data={len(self.data)}"


class Note:
    def __init__(self, pitch: str, octave: int, duration_symbol: str, tempo: str):
        """
        Initializes a Note object which combines musical properties to create sound.

        :param pitch: The musical pitch of the note.
        :param octave: The octave number of the note.
        :param duration_symbol: Symbol representing the duration of the note.
        :param tempo: The tempo at which the note is played.
        """
        self.pitch = pitch
        self.octave = octave
        self.duration_symbol = duration_symbol
        self.tempo = TEMPOS[tempo]
        self.frequency = self.calculate_frequency() if pitch else 0
        self.duration = self.calculate_duration()
        self.wave = Wave(0, self.duration) if not pitch else Wave(self.frequency, self.duration)

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

    def __str__(self) -> str:
        """
        Returns a string representation of the Note object.
        
        :returns: A string representation of the Note object.
        """
        return f"Note(pitch={self.pitch}, octave={self.octave}, duration={self.duration_symbol}, tempo={self.tempo})"

class Piano:
    def __init__(self):
        """
        Initializes the Piano which manages two lists for left and right hand notes.
        """
        self.notes_left = []
        self.notes_right = []
        self.time_left = 0 
        self.time_right = 0

    def add_note_left(self, note: Note) -> None:
        """
        Adds a note to the left hand's list of notes.

        :param note: The note to add.
        """
        self.notes_left.append((note, self.time_left))
        self.time_left += note.duration

    def add_note_right(self, note: Note) -> None:
        """
        Adds a note to the right hand's list of notes.

        :param note: The note to add.
        """
        self.notes_right.append((note, self.time_right))
        self.time_right += note.duration

    def get_combined_wave_array(self) -> np.ndarray:
        """
        Combines all notes from both hands into a single wave array.

        :return: A numpy array representing the combined wave of all notes.
        """
        max_duration = max(self.time_left, self.time_right)
        final_wave = Wave(0, max_duration, np.zeros(int(SAMPLERATE * max_duration)))

        final_wave = self.combine_notes(final_wave, self.notes_left)
        final_wave = self.combine_notes(final_wave, self.notes_right)

        return final_wave.data

    def combine_notes(self, final_wave: Wave, notes: List[Tuple[Note, float]]) -> Wave:
        """
        Combines notes from two hands together into a single wave.
        Each hand has the same total duration.
        This function utilizes magic method __add__() of Wave class to combine waves.
        
        :param final_wave: The initial wave to start combining into.
        :param notes: A list of notes along with their start times.
        :return: A wave object representing the combination of all input notes.
        """
        current_time = 0
        for note, start_time in notes:
            silence_duration = max(0, start_time - current_time)
            silence_wave = Wave(0, silence_duration, np.zeros(int(SAMPLERATE * silence_duration)))
            final_wave += silence_wave

            note_wave_extended = np.zeros_like(final_wave.data)
            start_index = int(SAMPLERATE * start_time)
            note_wave_extended[start_index:start_index + len(note.wave.data)] = note.wave.data
            note_wave_to_add = Wave(note.frequency, note.duration, note_wave_extended)
            final_wave += note_wave_to_add
            current_time = start_time + note.duration
        return final_wave
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Piano object.
        
        :returns: A string representation of the Piano object.
        """
        return f"Piano(notes_left={self.notes_left}, notes_right={self.notes_right})" 

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
        Loads the song data from a file, processes each line (for right and left hand) 
        to add notes to the each hand on piano, and plays the song by writing the combined wave to a WAV file.
        """
        file = open(self.filepath, 'r')
        tempos = file.readline().strip().split(',')
        tempo_user = input(f"Select a tempo for {self.filepath[:-4]}: {', '.join(tempos)}\n")
        while tempo_user not in tempos:
            tempo_user = input(f"Invalid tempo. Select a tempo for {self.filepath[:-4]}: {', '.join(tempos)}\n")
        lines = file.readlines()
        
        for i in range(0, len(lines), 3):
            right_hand_notes = lines[i + 1].strip().split('-')
            left_hand_notes = lines[i + 2].strip().split('-')
            
            self.process_notes(right_hand_notes, tempo_user, 'r')
            self.process_notes(left_hand_notes, tempo_user, 'l')

        write(self.filename_output, SAMPLERATE, self.piano.get_combined_wave_array().astype(np.int16))
        print(f"Song played and saved to {self.filename_output}")

    def process_notes(self, note_infos: List[str], tempo: str, hand: str):
        """
        Processes each note information string to create and add notes to the each hand on piano.

        :param note_infos: List of note information strings.
        :param tempo: The selected tempo for the notes.
        :param hand: Specifies whether the notes are for the left or right hand.
        """
        for note_info in note_infos:
            if note_info:
                note, duration = note_info.split(',')
                pitch = note[0] if len(note) > 1 else None
                octave = int(note[1]) if pitch else 0
                note = Note(pitch, octave, duration, tempo)
                if hand == 'r':
                    self.piano.add_note_right(note)
                elif hand == 'l':
                    self.piano.add_note_left(note)
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Song object.
        
        :returns: A string representation of the Song object.
        """
        return f"Song(filepath={self.filepath})"
    
def main():
    """
    Main function to load and play songs listed.
    """
    songs = ['alouette2.txt', 'row_row2.txt', 'twinkle_twinkle2.txt']
    for song in songs:
        my_song = Song(filepath=song)
        my_song.load_and_play_song()

if __name__ == "__main__":
    main()