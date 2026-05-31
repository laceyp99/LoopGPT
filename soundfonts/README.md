# SoundFonts Directory

This directory contains SoundFont (`.sf2`) files used for MIDI audio playback.

## Bundled Default

The project ships with `FM-Piano1 20190916.sf2` as the default bundled SoundFont.

- Source: https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html
- Upstream instrument: `FM-Piano1`
- Variant: small size, bright sound variation
- License: Creative Commons CC0 1.0

This keeps audio playback setup smaller than the older Salamander-based flow while still giving the app a piano SoundFont out of the box.

## Adding More SoundFonts

Any compatible `.sf2` file can live in this directory. Later UI changes will allow switching between multiple installed SoundFonts from the app.

Examples that the current audio module will auto-detect include:

- `SalamanderGrandPiano.sf2`
- `salamander-grand-piano.sf2`
- `piano.sf2`
- `GeneralUser.sf2`
- `FluidR3_GM.sf2`
- Any other `.sf2` file in this directory

## Current Limits

The app still requires FluidSynth and FFmpeg to be installed separately for audio rendering. Bundling the default SoundFont only removes the manual SoundFont download step.
