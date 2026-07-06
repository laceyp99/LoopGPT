"""Synchronous prompt-to-loop generation engine."""

from collections.abc import Callable
import json
import os

from mido import MidiFile

from conductor_core import music
from conductor_core import playback
from conductor_core import routing
from conductor_core.config import (
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    ProgressEvent,
)
from conductor_core.midi import loop_to_midi
from conductor_core.storage import FilesystemArtifactStore, get_provider_for_model

ProgressCallback = Callable[[ProgressEvent], None]


class LoopGenerationEngine:
    """Synchronous Core engine for provider-backed loop generation."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        store: FilesystemArtifactStore | None = None,
    ):
        self.config = config or EngineConfig.from_defaults()
        self.store = store or FilesystemArtifactStore(self.config.artifact_root)

    def _emit(
        self,
        progress_callback: ProgressCallback | None,
        stage: str,
        message: str,
        detail: str | None = None,
    ) -> None:
        if progress_callback:
            progress_callback(ProgressEvent(stage=stage, message=message, detail=detail))

    def generate(
        self,
        request: GenerationRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> GenerationResult:
        """Generate a loop and persist MIDI/messages/metadata artifacts."""
        workspace = None
        finalized = False
        warnings = []
        prompt = f"{request.key} {request.scale} {request.description}."
        system_prompt = (
            request.prompt_override
            or self.config.prompt_override
            or music.get_loop_prompt()
        )

        try:
            self._emit(progress_callback, "provider_call", "Generating MIDI...")
            loop, messages, total_cost = routing.generate_midi(
                model_choice=request.model,
                prompt=prompt,
                temp=request.temperature,
                use_thinking=request.use_thinking,
                effort=request.effort,
                provider_credentials=self.config.provider_credentials,
                system_prompt=system_prompt,
            )

            self._emit(progress_callback, "midi", "Processing MIDI...")
            workspace = self.store.create_generation_workspace()
            midi = MidiFile()
            model_info = music.get_model_info()
            loop_to_midi(
                midi,
                loop,
                times_as_string=request.model in model_info["models"]["Google"].keys(),
            )
            midi.save(workspace.midi_path)

            audio_path = None
            selected_soundfont = request.soundfont_path or self.config.default_soundfont_path
            if request.render_audio:
                self._emit(progress_callback, "audio", "Rendering Audio...")
                audio_path = playback.midi_to_mp3(
                    workspace.midi_path,
                    output_path=workspace.audio_path,
                    soundfont_name=str(selected_soundfont) if selected_soundfont else None,
                )
                if audio_path is None:
                    warnings.append("Audio rendering was skipped or failed.")

            with open(workspace.messages_path, "w", encoding="utf-8") as messages_file:
                json.dump(messages, messages_file, indent=2)

            provider = request.provider or get_provider_for_model(request.model, model_info)
            metadata = self.store.finalize_generation(
                workspace=workspace,
                prompt=request.description,
                key=request.key,
                scale=request.scale,
                model=request.model,
                provider=provider,
                temperature=request.temperature,
                cost=total_cost,
                soundfont=os.path.basename(str(selected_soundfont))
                if audio_path and selected_soundfont
                else None,
            )
            finalized = True
            return GenerationResult(
                generation_id=metadata.id,
                loop=loop,
                midi_path=metadata.midi_path,
                audio_path=metadata.audio_path,
                messages=messages,
                cost=total_cost,
                metadata=metadata,
                warnings=warnings,
            )
        finally:
            if workspace is not None and not finalized:
                self.store.cleanup_generation_workspace(workspace)
