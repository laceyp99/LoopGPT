"""Public configuration and request/result contracts for Conductor Core."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderCredentials:
    """Provider credentials supplied by an app or script."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    ollama_host: str | None = None


@dataclass(frozen=True)
class EngineConfig:
    """Environment, resource, and storage configuration for the core engine."""

    artifact_root: str | Path = "generations"
    provider_credentials: ProviderCredentials = field(default_factory=ProviderCredentials)
    prompt_override: str | None = None
    default_soundfont_path: str | Path | None = None

    @classmethod
    def from_defaults(
        cls,
        artifact_root: str | Path = "generations",
        provider_credentials: ProviderCredentials | None = None,
        prompt_override: str | None = None,
        default_soundfont_path: str | Path | None = None,
    ) -> "EngineConfig":
        """Create a config using Core defaults plus caller-provided overrides."""
        return cls(
            artifact_root=artifact_root,
            provider_credentials=provider_credentials or ProviderCredentials(),
            prompt_override=prompt_override,
            default_soundfont_path=default_soundfont_path,
        )


@dataclass(frozen=True)
class GenerationRequest:
    """One prompt-to-loop generation request."""

    key: str
    scale: str
    description: str
    model: str
    provider: str | None = None
    temperature: float = 0.0
    use_thinking: bool = False
    effort: str = "low"
    prompt_override: str | None = None
    render_audio: bool = False
    soundfont_path: str | Path | None = None


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress event emitted by the synchronous engine."""

    stage: str
    message: str
    detail: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Complete result of a generated loop and its persisted artifacts."""

    generation_id: str
    loop: Any
    midi_path: str
    audio_path: str | None
    messages: list[dict[str, Any]]
    cost: float | None
    metadata: Any
    warnings: list[str] = field(default_factory=list)
