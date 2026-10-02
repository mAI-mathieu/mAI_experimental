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
    roots = model_roots()
    for root in roots:
        directory = (root / name).resolve()
        if directory.is_relative_to(root) and directory.is_dir():
            # Validate the selected folder directly so incomplete downloads and
            # invalid metadata report their real error instead of "not found".
            return validate_package(directory)
    searched = "\n".join(f"  {root / name / 'model_index.json'}" for root in roots)
    raise FileNotFoundError(
        f"SoL H3 refiner package '{name}' was not found. Searched:\n{searched}\n"
        f"Download the complete {DEFAULT_MODEL} package into models/sol_refiner/{DEFAULT_FOLDER}/ "
        "or a configured sol_refiner/diffusers model root. MiniMax H3 generation weights are a separate model. "
        "The node never downloads weights."
    )
