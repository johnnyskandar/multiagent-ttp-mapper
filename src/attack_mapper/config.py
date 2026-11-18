"""Central configuration: environment variables, constants, and project paths.

All paths are built from the project root so the repo works from any clone location.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# src/attack_mapper/config.py -> project root is two levels above the package dir
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


# LLM and embeddings
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "").strip()
LLM_CACHE: bool = _env_bool("LLM_CACHE", True)

# Limits
MAX_REPORT_CHARS: int = _env_int("MAX_REPORT_CHARS", 20000)
MIN_REPORT_CHARS: int = 50
MAX_RETRIES: int = 2

# Paths
DATA_DIR: Path = PROJECT_ROOT / "data"
ATTACK_DATA_DIR: Path = DATA_DIR / "attack"
CISA_DIR: Path = DATA_DIR / "cisa"
DEMO_DATA_DIR: Path = DATA_DIR / "demo"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
LLM_CACHE_DIR: Path = PROJECT_ROOT / "llm_cache"
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"
EVAL_DIR: Path = PROJECT_ROOT / "eval"
EVAL_DATA_DIR: Path = EVAL_DIR / "data"
EVAL_RESULTS_DIR: Path = EVAL_DIR / "results"
EVAL_RUNS_DIR: Path = EVAL_DIR / "runs"

# Chroma collection names
TECHNIQUES_COLLECTION: str = "attack_techniques"
MITIGATIONS_COLLECTION: str = "attack_mitigations"
