from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline


def __getattr__(name: str):
    """Lazy-import text pipeline so SS-only code paths do not require open3d."""
    if name == "TrellisTextTo3DPipeline":
        from .trellis_text_to_3d import TrellisTextTo3DPipeline as _TrellisTextTo3DPipeline

        return _TrellisTextTo3DPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _pipeline_cls(name: str):
    if name == "TrellisTextTo3DPipeline":
        from .trellis_text_to_3d import TrellisTextTo3DPipeline

        return TrellisTextTo3DPipeline
    try:
        return globals()[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline class {name!r}") from exc


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import json
    import os

    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    cls = _pipeline_cls(config["name"])
    return cls.from_pretrained(path)
