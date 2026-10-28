"""Load repository-local configuration without displaying credentials."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """LLM configuration; credentials are excluded from repr and str."""

    llm_backend: Literal["meta", "baseten"] = "meta"
    model_api_key: str = field(default="", repr=False)
    muse_model: str = ""
    baseten_api_key: str = field(default="", repr=False)
    baseten_base_url: str = ""
    baseten_model: str = ""


def load_settings(env_path: Path | None = None) -> Settings:
    """Load .env without overriding process environment or searching parents."""
    path = env_path if env_path is not None else Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=path, override=False)
    backend = os.getenv("LLM_BACKEND", "meta")
    if backend not in ("meta", "baseten"):
        raise ValueError("LLM_BACKEND must be meta or baseten")
    return Settings(
        llm_backend=backend,
        model_api_key=os.getenv("MODEL_API_KEY", ""),
        muse_model=os.getenv("MUSE_MODEL", ""),
        baseten_api_key=os.getenv("BASETEN_API_KEY", ""),
        baseten_base_url=os.getenv("BASETEN_BASE_URL", ""),
        baseten_model=os.getenv("BASETEN_MODEL", ""),
    )


settings = load_settings()
