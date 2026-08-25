from pathlib import Path


PROFILES_DIRECTORY = Path(__file__).resolve().parents[1] / "profiles"

KROMA_PROFILE_DIRECTORY = PROFILES_DIRECTORY / "kroma"

KROMA_BALANCED_PROFILE_PATH = (
    KROMA_PROFILE_DIRECTORY
    / "kroma-v0.1-balanced.json"
)

ILLUSTRIOUS_PROFILE_DIRECTORY = PROFILES_DIRECTORY / "illustrious"

ILLUSTRIOUS_COMBINED_PROFILE_PATH = (
    ILLUSTRIOUS_PROFILE_DIRECTORY
    / "illustrious-combined-int8-mixed-convrot-clip-l.json"
)
