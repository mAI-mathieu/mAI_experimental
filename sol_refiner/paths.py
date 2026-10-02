"""Discover complete local packages through ComfyUI model folder configuration."""

from pathlib import Path

from .utils import DEFAULT_FOLDER, DEFAULT_MODEL, read_json, validate_package


def model_roots():
    import folder_paths

    folder_paths.add_model_folder_path("sol_refiner", str(Path(folder_paths.models_dir) / "sol_refiner"))
    roots = folder_paths.get_folder_paths("sol_refiner")
    if "diffusers" in folder_paths.folder_names_and_paths:
        roots += folder_paths.get_folder_paths("diffusers")
    return list(dict.fromkeys(Path(root).resolve() for root in roots))


def discovered_models():
    models = {}
    for root in model_roots():
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not (directory / "model_index.json").is_file():
                continue
            if not directory.resolve().is_relative_to(root):
                continue
            try:
                index = read_json(directory / "model_index.json")
            except ValueError:
                continue  # Unrelated/broken folders are not selections; explicit loading validates again.
            if isinstance(index, dict) and index.get("_class_name") == "SoLRefinerH3Pipeline":
                models.setdefault(directory.name, directory.resolve())
    return models


def model_names():
    return [DEFAULT_MODEL, *sorted(name for name in discovered_models() if name != DEFAULT_MODEL)]


def resolve_model(name):
    if name == DEFAULT_MODEL:
        name = DEFAULT_FOLDER
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
        raise ValueError("Select a local SoL model folder from the loader list.")
    models = discovered_models()
    if name not in models:
        raise FileNotFoundError(f"SoL H3 model '{name}' was not found. Download the complete package into models/sol_refiner/{DEFAULT_FOLDER}/ or a configured sol_refiner/diffusers model root. The node never downloads weights.")
    return validate_package(models[name])
