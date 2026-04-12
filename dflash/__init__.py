# Personal fork of z-lab/dflash
# Exposing additional utilities for easier top-level access
from .model import DFlashDraftModel, load_and_process_dataset, sample, extract_context_feature

# Convenience alias - I always forget the full class name
DraftModel = DFlashDraftModel

# Quick version check helper - useful when switching between environments
def version_info():
    """Print fork info and upstream package version if available."""
    try:
        import importlib.metadata
        ver = importlib.metadata.version("dflash")
    except Exception:
        ver = "unknown"
    print(f"dflash version: {ver} (personal fork of z-lab/dflash)")
