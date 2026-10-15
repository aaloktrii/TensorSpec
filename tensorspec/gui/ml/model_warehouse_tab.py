"""Model Warehouse: catalog-driven list/load for local, torch.hub, pip, and HF models."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

CATALOG_PATH = Path(__file__).resolve().parent / "model_catalog.json"

REQUIRED_FIELDS = (
    "name",
    "category",
    "description",
    "paper_url",
    "code_url",
    "license",
    "loader",
)


def load_model_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or CATALOG_PATH
    with open(catalog_path, encoding="utf-8") as fh:
        return json.load(fh)


def _pip_install_hint(entry: dict[str, Any]) -> str:
    if entry.get("install_cmd"):
        return str(entry["install_cmd"])
    package = (entry.get("package") or "").strip()
    if package:
        return f"pip install {package}"
    return "See code_url for install instructions."


class ModelLoadWorker(QThread):
    """Download / import a catalog entry off the GUI thread."""

    finished_ok = Signal(str, object)
    finished_err = Signal(str)

    def __init__(self, entry: dict[str, Any], parent=None):
        super().__init__(parent)
        self.entry = entry

    def run(self):
        try:
            model = self._load(self.entry)
            self.finished_ok.emit(self.entry["name"], model)
        except Exception as exc:  # noqa: BLE001 — surface any loader failure to UI
            self.finished_err.emit(str(exc))

    def _load(self, entry: dict[str, Any]):
        loader = entry.get("loader")
        if loader == "local":
            return {"type": "local", "name": entry["name"]}

        if entry.get("loadable") is False:
            raise RuntimeError(
                entry.get("notes")
                or "This model is catalogued for reference only (not loadable)."
            )

        if loader == "torch_hub":
            import torch

            cache_dir = "tensorspec/gui/ml/model_warehouse/pretrained"
            os.makedirs(cache_dir, exist_ok=True)
            torch.hub.set_dir(cache_dir)
            return torch.hub.load(entry["repo"], entry["model_name"])

        if loader == "pip":
            return self._load_pip(entry)

        if loader == "huggingface":
            return self._load_huggingface(entry)

        raise RuntimeError(f"Unknown loader type: {loader!r}")

    def _load_pip(self, entry: dict[str, Any]):
        package = (entry.get("package") or "").strip()
        import_name = (entry.get("import_name") or "").strip()
        snippet = (entry.get("import_snippet") or "").strip()
        hint = _pip_install_hint(entry)

        if not package and not import_name and not snippet:
            raise RuntimeError(
                f"{entry['name']} has no pip package. Install from source:\n{hint}"
            )

        module_to_try = import_name or package.replace("-", "_")
        try:
            if snippet:
                # Execute a short import smoke-test in an isolated namespace.
                ns: dict[str, Any] = {}
                exec(snippet, ns, ns)  # noqa: S102 — curated catalog snippets only
                return {"type": "pip", "name": entry["name"], "namespace_keys": list(ns)}
            importlib.import_module(module_to_try)
            return {"type": "pip", "name": entry["name"], "module": module_to_try}
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Package not installed for {entry['name']}.\n\n"
                f"Install with:\n  {hint}\n\n"
                f"Details: {exc}"
            ) from exc

    def _load_huggingface(self, entry: dict[str, Any]):
        hf_repo = entry.get("hf_repo")
        if not hf_repo:
            raise RuntimeError(f"{entry['name']}: missing hf_repo in catalog.")
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Package not installed for Hugging Face loading.\n\n"
                "Install with:\n  pip install transformers\n\n"
                f"Details: {exc}"
            ) from exc
        return AutoModel.from_pretrained(hf_repo)


class ModelWarehouseTab(QWidget):
    def __init__(self, parent_app=None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.catalog = load_model_catalog()
        self._models_by_name = {m["name"]: m for m in self.catalog.get("models", [])}
        self._worker: Optional[ModelLoadWorker] = None
        self._loaded: dict[str, Any] = {}
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderHidden(True)
        self.model_tree.setSelectionMode(QTreeWidget.SingleSelection)
        self._populate_tree()
        self.model_tree.itemSelectionChanged.connect(self.on_model_selected)
        main_layout.addWidget(self.model_tree, 1)

        right_layout = QVBoxLayout()
        self.properties_text = QTextEdit()
        self.properties_text.setReadOnly(True)
        self.properties_text.setPlaceholderText("Select a model to view properties...")
        right_layout.addWidget(self.properties_text, 1)

        self.action_button = QPushButton("Load")
        self.action_button.setEnabled(False)
        self.action_button.clicked.connect(self.on_action_clicked)
        right_layout.addWidget(self.action_button)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        main_layout.addLayout(right_layout, 2)

    def _populate_tree(self):
        self.model_tree.clear()
        models = self.catalog.get("models", [])
        order = self.catalog.get("category_order") or []
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for entry in models:
            by_cat.setdefault(entry["category"], []).append(entry)

        seen = set()
        categories = list(order) + [c for c in by_cat if c not in order]
        for category in categories:
            entries = by_cat.get(category)
            if not entries:
                continue
            seen.add(category)
            parent = QTreeWidgetItem([category])
            parent.setFlags(Qt.ItemIsEnabled)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            for entry in entries:
                child = QTreeWidgetItem([entry["name"]])
                child.setData(0, Qt.UserRole, entry["name"])
                parent.addChild(child)
            self.model_tree.addTopLevelItem(parent)
            parent.setExpanded(category in ("Local Models", "Spectroscopy / image"))

    def _selected_entry(self) -> Optional[dict[str, Any]]:
        items = self.model_tree.selectedItems()
        if not items:
            return None
        name = items[0].data(0, Qt.UserRole)
        if not name:
            return None
        return self._models_by_name.get(name)

    def on_model_selected(self):
        entry = self._selected_entry()
        if not entry:
            self.properties_text.clear()
            self.action_button.setEnabled(False)
            self.status_label.setText("")
            return

        lines = [
            f"Model: {entry['name']}",
            f"Category: {entry['category']}",
            f"Loader: {entry['loader']}",
            f"License: {entry.get('license') or '(none)'}",
            f"Verified: {entry.get('verified', False)}",
            "",
            "Description:",
            entry.get("description") or "(none)",
        ]
        if entry.get("paper_url"):
            lines += ["", f"Paper: {entry['paper_url']}"]
        if entry.get("code_url"):
            lines += [f"Code: {entry['code_url']}"]
        if entry.get("loader") == "pip":
            lines += ["", f"Install: {_pip_install_hint(entry)}"]
        if entry.get("loader") == "torch_hub":
            lines += [
                "",
                f"torch.hub: {entry.get('repo')} / {entry.get('model_name')}",
            ]
        if entry.get("loader") == "huggingface":
            lines += ["", f"Hugging Face: {entry.get('hf_repo')}"]
        if entry.get("notes"):
            lines += ["", f"Notes: {entry['notes']}"]

        self.properties_text.setText("\n".join(lines))

        if entry.get("loader") == "local":
            self.action_button.setText("Load")
            self.action_button.setEnabled(True)
            self.status_label.setText("Status: Ready to load")
        elif entry.get("loadable") is False:
            self.action_button.setText("Unavailable")
            self.action_button.setEnabled(False)
            self.status_label.setText("Status: Code not released / not loadable")
        else:
            self.action_button.setText("Download / Load")
            self.action_button.setEnabled(True)
            self.status_label.setText("Status: Ready")

    def on_action_clicked(self):
        entry = self._selected_entry()
        if not entry:
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "A model load is already running.")
            return

        if entry.get("loader") == "local":
            self.status_label.setText(f"Status: Loaded {entry['name']} locally.")
            self._loaded[entry["name"]] = {"type": "local", "name": entry["name"]}
            return

        self.action_button.setEnabled(False)
        self.status_label.setText(f"Status: Loading {entry['name']}…")
        self._worker = ModelLoadWorker(entry, parent=self)
        self._worker.finished_ok.connect(self._on_load_ok)
        self._worker.finished_err.connect(self._on_load_err)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_load_ok(self, name: str, model: object):
        self._loaded[name] = model
        self.status_label.setText(f"Status: Successfully loaded {name}")
        QMessageBox.information(self, "Success", f"{name} loaded successfully.")

    def _on_load_err(self, message: str):
        self.status_label.setText("Status: Error loading model")
        QMessageBox.critical(self, "Error", message)

    def _on_worker_finished(self):
        self.action_button.setEnabled(self._selected_entry() is not None)
        self._worker = None
