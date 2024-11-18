'''
This file holds all the objects that will be used to generate MIDI information.

This includes the following objects:
* SixteenthNote: An enumeration of the 16 possible sixteenth notes in a bar
* TimeInformation: A class that holds the start beat and duration of a chord or note
* Note: A class that holds the pitch, octave, and velocity of a note
* Chord: A class that holds the root note, quality, a list of note voicings, and time information of a chord
* Bar: A class that holds a list of chords in the bar
* MelodyNote: A class that holds the pitch, octave, velocity, and time information of a melody note
* MelodyBar: A class that holds a list of melody notes in the bar
'''

# IMPORTS
from pydantic import BaseModel, field_validator, Field
from enum import IntEnum

class SixteenthNote(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    ELEVEN = 11
    TWELVE = 12
    THIRTEEN = 13
    FOURTEEN = 14
    FIFTEEN = 15
    SIXTEEN = 16
   
class TimeInformation(BaseModel):
    start_beat: SixteenthNote = Field(..., description="Starting beat of the chord in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.")  # Start beat of the chord (e.g. 1-8)
    duration: SixteenthNote = Field(..., description="Duration of the chord in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.")  # Duration of the chord (e.g. 1-4)

class Note(BaseModel):
    pitch: str = Field(..., description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B")')
    octave: int = Field(..., description='Octave of the note (e.g. 1-8)')
    velocity: int = Field(..., description='Velocity of the note (e.g. 0-127)') 

class Chord(BaseModel):
    root: str = Field(..., description='Root note of the chord (e.g. "C", "D", "E", "F", "G", "A", "B")')
    quality: str = Field(..., description='Quality of the chord (e.g. "major", "minor", "diminished", "augmented", "dominant")')
    voicing: list[Note] = Field(..., description='List of notes in the chord')
    time: TimeInformation

class Bar(BaseModel):
    num: int = Field(..., description='Number of the bar (e.g. 1-4)')
    chords: list[Chord] = Field(..., description='List of chords in the bar')
    
    @field_validator('chords')
    def check_total_sixteenth_notes(cls, chords):
        total_duration = sum(chord.time.duration for chord in chords)
        if total_duration > 16:
            raise ValueError(f'Total duration of chords in the bar must not exceed 16 beats, but got {total_duration}')
        return chords
    
class MelodyNote(BaseModel):
    pitch: str = Field(..., description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B")')
    octave: int = Field(..., description='Octave of the note (e.g. 1-8)')
    velocity: int = Field(..., description='Velocity of the note (e.g. 0-127)')
    time: TimeInformation 

class MelodyBar(BaseModel):
    num: int = Field(..., description='Number of the bar (e.g. 1-4)')
    melody: list[MelodyNote] = Field(..., description='List of notes in the bar')
    
    @field_validator('melody')
    def check_total_sixteenth_notes(cls, melody):
        total_duration = sum(note.time.duration for note in melody)
        if total_duration > 16:
            raise ValueError(f'Total duration of melody in the bar must not exceed 16 beats, but got {total_duration}')
        return melody