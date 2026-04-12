# Personal fork of z-lab/dflash
# Exposing additional utilities for easier top-level access
from .model import DFlashDraftModel, load_and_process_dataset, sample, extract_context_feature

# Convenience alias - I always forget the full class name
DraftModel = DFlashDraftModel
