"""Compatibility exports for MIDI checks now owned by :mod:`conductor_eval`."""

from conductor_eval.checks import (
    arpeggio_test,
    duration_test,
    is_monophonic,
    polyphonic_profile,
    run_midi_tests,
    scale_test,
)

__all__ = [
    "arpeggio_test",
    "duration_test",
    "is_monophonic",
    "polyphonic_profile",
    "run_midi_tests",
    "scale_test",
]
