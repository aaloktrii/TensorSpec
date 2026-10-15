"""Schema checks for the Model Warehouse catalog."""

from pathlib import Path

from tensorspec.gui.ml.model_warehouse_tab import (
    CATALOG_PATH,
    REQUIRED_FIELDS,
    load_model_catalog,
)

VALID_LOADERS = {"local", "torch_hub", "pip", "huggingface"}


def test_model_catalog_exists():
    assert CATALOG_PATH.is_file()


def test_model_catalog_entries_have_required_fields():
    catalog = load_model_catalog()
    models = catalog.get("models")
    assert isinstance(models, list) and models, "catalog must list models"

    for entry in models:
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        assert not missing, f"{entry.get('name', '<unnamed>')}: missing {missing}"
        assert entry["loader"] in VALID_LOADERS, (
            f"{entry['name']}: invalid loader {entry['loader']!r}"
        )
        assert isinstance(entry["name"], str) and entry["name"].strip()
        assert isinstance(entry["category"], str) and entry["category"].strip()
        assert isinstance(entry["description"], str)

        loader = entry["loader"]
        if loader == "torch_hub":
            assert entry.get("repo") and entry.get("model_name"), entry["name"]
        elif loader == "huggingface":
            assert entry.get("hf_repo"), entry["name"]
        elif loader == "pip":
            # Either a PyPI package or an explicit install_cmd / non-loadable note.
            has_pkg = bool((entry.get("package") or "").strip())
            has_cmd = bool((entry.get("install_cmd") or "").strip())
            non_loadable = entry.get("loadable") is False
            assert has_pkg or has_cmd or non_loadable, entry["name"]


def test_model_catalog_keeps_dinov2_and_local_models():
    catalog = load_model_catalog()
    names = {m["name"] for m in catalog["models"]}
    for required in (
        "SimpleCAE",
        "DINOv2 ViT-S/14",
        "DINOv2 ViT-B/14",
        "DINOv2 ViT-L/14",
        "DINOv2 ViT-g/14",
        "MACE-MP-0",
        "O-kRystal",
    ):
        assert required in names

    ok_crystal = next(m for m in catalog["models"] if m["name"] == "O-kRystal")
    assert ok_crystal.get("loadable") is False
    assert "code not released" in (ok_crystal.get("notes") or "").lower() or (
        "code not released" in (ok_crystal.get("license") or "").lower()
    )


def test_model_catalog_path_is_under_gui_ml():
    assert CATALOG_PATH == Path(__file__).resolve().parents[1] / (
        "tensorspec/gui/ml/model_catalog.json"
    )
