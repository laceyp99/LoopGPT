'''
This file holds all the objects that will be used to generate MIDI information.

Note: Gemini structured outputs do not support integer enums. To work around this limitation, _G objects are used instead.
'''

from pydantic import BaseModel, Field
from enum import IntEnum, Enum

# Sixteenth Note Objects
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
    
class SixteenthNote_G(Enum):
    ONE = "one"
    TWO = "two"
    THREE = "three"
    FOUR = "four"
    FIVE = "five"
    SIX = "six"
    SEVEN = "seven"
    EIGHT = "eight"
    NINE = "nine"
    TEN = "ten"
    ELEVEN = "eleven"
    TWELVE = "twelve"
    THIRTEEN = "thirteen"
    FOURTEEN = "fourteen"
    FIFTEEN = "fifteen"
    SIXTEEN = "sixteen"

# Time Information Objects
class TimeInformation(BaseModel):
    start_beat: SixteenthNote = Field(..., description="Starting beat of the chord in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.")  # Start beat of the chord (e.g. 1-8)
    duration: SixteenthNote = Field(..., description="Duration of the chord in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.")  # Duration of the chord (e.g. 1-4)

class TimeInformation_G(BaseModel):
    start_beat: SixteenthNote_G = Field(..., description="Starting beat of the chord in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.")  # Start beat of the chord (e.g. 1-8)
    duration: SixteenthNote_G = Field(..., description="Duration of the chord in sixteenth notes (e.g. 1-16). REMEMBER THIS IS BASE 1 NOT 0.")  # Duration of the chord (e.g. 1-4)

# Note Objects
class Note(BaseModel):
    pitch: str = Field(..., description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B")')
    octave: int = Field(..., description='Octave of the note (e.g. 1-8)')
    velocity: int = Field(..., description='Velocity of the note (e.g. 0-127)')
    time: TimeInformation

class Note_G(BaseModel):
    pitch: str = Field(..., description='Pitch of the note (e.g. "C", "D", "E", "F", "G", "A", "B")')
    octave: int = Field(..., description='Octave of the note (e.g. 1-8)')
    velocity: int = Field(..., description='Velocity of the note (e.g. 0-127)')
    time: TimeInformation_G = Field(..., description='Time information of the note')

# Bar Objects
class Bar(BaseModel):
    num: int = Field(..., description='Number of the bar (e.g. 1-4)')
    notes: list[Note] = Field(..., description='List of notes in the bar')

class Bar_G(BaseModel):
    num: int = Field(..., description='Number of the bar (e.g. 1-4)')
    notes: list[Note_G] = Field(..., description='List of notes in the bar')

# Loop Objects
class Loop(BaseModel):
    Bar_1: Bar = Field(..., description='The first bar of the four bar loop')
    Bar_2: Bar = Field(..., description='The second bar of the four bar loop')
    Bar_3: Bar = Field(..., description='The third bar of the four bar loop')
    Bar_4: Bar = Field(..., description='The fourth bar of the four bar loop')

class Loop_G(BaseModel):
    Bar_1: Bar_G = Field(..., description='The first bar of the four bar loop')
    Bar_2: Bar_G = Field(..., description='The second bar of the four bar loop')
    Bar_3: Bar_G = Field(..., description='The third bar of the four bar loop')
    Bar_4: Bar_G = Field(..., description='The fourth bar of the four bar loop')