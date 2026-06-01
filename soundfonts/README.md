# SoundFonts Directory

This directory contains SoundFont (`.sf2`) files used for MIDI audio playback.

## Bundled Default

The project ships with `FM-Piano1 20190916.sf2` as the default bundled SoundFont.

- Source: https://freepats.zenvoid.org/ElectricPiano/synthesized-piano.html
- Upstream instrument: `FM-Piano1`
- Type: compact synthesized piano
- License: Creative Commons CC0 1.0

This keeps audio playback setup smaller than the older Salamander-based flow while still giving the app a piano SoundFont out of the box.

## Adding More SoundFonts

Any compatible `.sf2` file can live in this directory.

Current app behavior:

- The Gradio app lists installed `.sf2` files in the **SoundFont** dropdown.
- If you add a new SoundFont while the app is already running, click **Refresh SoundFonts** to reload the directory.
- Use **Re-render Audio** to audition the currently loaded MIDI with a different SoundFont without regenerating the loop.
- Saved history entries remember which SoundFont produced their stored audio preview.

Examples that the current audio module will auto-detect include:

- `SalamanderGrandPiano.sf2`
- `salamander-grand-piano.sf2`
- `piano.sf2`
- `GeneralUser.sf2`
- `FluidR3_GM.sf2`
- Any other `.sf2` file in this directory

## Current Limits

The app still requires FluidSynth and FFmpeg to be installed separately for audio rendering. Bundling the default SoundFont only removes the manual SoundFont download step.
