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
    # Also show Python version for easier debugging across envs
    import sys
    py_ver = sys.version.split()[0]
    print(f"dflash version: {ver} (personal fork of z-lab/dflash) | Python {py_ver}")

# Return version string instead of just printing - makes it easier to use
# programmatically (e.g. in notebooks or assertion checks)
def get_version():
    """Return the dflash version string, or 'unknown' if not found."""
    try:
        import importlib.metadata
        return importlib.metadata.version("dflash")
    except Exception:
        return "unknown"

# NOTE: version_info() returns None (just prints), which is awkward in notebooks.
# Patching it here to also return the version string for convenience.
_original_version_info = version_info
def version_info():
    """Print fork info and upstream package version. Also returns the version string."""
    _original_version_info()
    return get_version()
