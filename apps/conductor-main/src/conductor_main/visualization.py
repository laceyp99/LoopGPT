"""Plotly piano-roll visualization owned by the Gradio client."""

import mido
import plotly.graph_objects as go


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
BACKGROUND = "#1a1a2e"
BLACK_KEY_BACKGROUND = "#252540"
GRID = "#2a2a4a"
BAR_GRID = "#4a4a6a"
TEXT = "#e0e0e0"


def _is_black_key(pitch):
    return pitch % 12 in {1, 3, 6, 8, 10}


def _velocity_color(velocity):
    amount = velocity / 127.0
    red = int(45 + amount * 210)
    green = int(27 + amount * 80)
    blue = int(78 + amount * 13)
    return f"rgb({red},{green},{blue})"


def _note_label(pitch):
    return NOTE_NAMES[pitch % 12], (pitch // 12) - 1


def _duration_label(sixteenths):
    labels = {1: "1/16", 2: "1/8", 4: "1/4", 8: "1/2", 16: "1 bar"}
    value = int(sixteenths)
    return labels.get(value, f"{value}/16")


def visualize_midi_plotly(input_midi):
    """Render a MIDI file or ``MidiFile`` as a four-bar piano roll."""
    if isinstance(input_midi, str):
        midi = mido.MidiFile(input_midi)
    elif isinstance(input_midi, mido.MidiFile):
        midi = input_midi
    else:
        raise ValueError("Input must be a filename or a MidiFile object")

    elapsed = 0
    active = {}
    notes = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += message.time
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault(message.note, []).append((elapsed, message.velocity))
        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            if active.get(message.note):
                start, velocity = active[message.note].pop(0)
                notes.append((message.note, start, elapsed, velocity))

    sixteenth_notes = [
        (pitch, start / midi.ticks_per_beat * 4, end / midi.ticks_per_beat * 4, velocity)
        for pitch, start, end, velocity in notes
    ]
    figure = go.Figure()
    if not sixteenth_notes:
        figure.add_annotation(
            text="No notes found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return figure

    pitches = [note[0] for note in sixteenth_notes]
    minimum = max(0, min(pitches) - 1)
    maximum = min(127, max(pitches) + 1)

    for pitch in range(minimum, maximum + 1):
        if _is_black_key(pitch):
            figure.add_shape(
                type="rect", x0=0, x1=64, y0=pitch - 0.5, y1=pitch + 0.5,
                fillcolor=BLACK_KEY_BACKGROUND, line=dict(width=0), layer="below",
            )
    for beat in range(1, 16):
        if beat % 4:
            position = beat * 4
            figure.add_shape(
                type="line", x0=position, x1=position, y0=minimum - 0.5, y1=maximum + 0.5,
                line=dict(color=GRID, width=1), layer="below",
            )
    for bar in range(5):
        position = bar * 16
        figure.add_shape(
            type="line", x0=position, x1=position, y0=minimum - 0.5, y1=maximum + 0.5,
            line=dict(color=BAR_GRID, width=2), layer="below",
        )
    for bar in range(1, 5):
        figure.add_annotation(
            x=(bar - 1) * 16 + 8, y=maximum + 1, text=f"Bar {bar}", showarrow=False,
            font=dict(color=TEXT, size=12), yanchor="bottom",
        )

    hover_x, hover_y, hover_text = [], [], []
    for pitch, start, end, velocity in sixteenth_notes:
        figure.add_shape(
            type="rect", x0=start, x1=end, y0=pitch - 0.4, y1=pitch + 0.4,
            fillcolor=_velocity_color(velocity),
            line=dict(color="rgba(255,255,255,0.3)", width=1), layer="above",
        )
        bar = int(start // 16) + 1
        beat = int((start % 16) // 4) + 1
        subdivision = int(start % 4) + 1
        name, octave = _note_label(pitch)
        hover_x.append((start + end) / 2)
        hover_y.append(pitch)
        hover_text.append(
            f"Note: {name}{octave}<br>Velocity: {velocity}<br>"
            f"Position: Bar {bar}, Beat {beat}.{subdivision}<br>"
            f"Duration: {_duration_label(end - start)}"
        )

    figure.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers", marker=dict(size=10, opacity=0),
        hoverinfo="text", hovertext=hover_text,
        hoverlabel=dict(bgcolor=GRID, font_size=12, font_color=TEXT),
    ))
    ticks = list(range(minimum, maximum + 1))
    labels = [f"{name}{octave}" for name, octave in map(_note_label, ticks)]
    figure.update_layout(
        plot_bgcolor=BACKGROUND, paper_bgcolor=BACKGROUND,
        xaxis=dict(range=[0, 64], showgrid=False, zeroline=False, tickmode="array",
                   tickvals=[0, 16, 32, 48, 64], ticktext=["", "", "", "", ""], fixedrange=False),
        yaxis=dict(range=[minimum - 0.5, maximum + 1.5], showgrid=False, zeroline=False,
                   tickmode="array", tickvals=ticks, ticktext=labels,
                   tickfont=dict(color=TEXT, size=10)),
        showlegend=False, margin=dict(l=60, r=20, t=40, b=40), height=400,
        hoverdistance=20, hovermode="closest",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TEXT, activecolor="#FF6B5B"),
    )
    return figure
