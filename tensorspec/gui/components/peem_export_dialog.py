"""Export options for one map of the XPEEM map tab: files, region, picture size, colours.

Front end only: it starts from ``PeemService.export_options`` and returns the keyword arguments of
``PeemService.export_map``; the service writes the files."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from tensorspec.gui.components.peem_map_common import _dspin, _numbers

CMAPS = ("bwr", "RdBu_r", "seismic", "coolwarm", "gray", "viridis", "magma", "inferno", "cividis")


def _ispin(lo: int, hi: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s


class ExportMapDialog(QDialog):
    """opts: PeemService.export_options(...); roi_box: (x0, x1, y0, y1) of the map ROI or None."""

    def __init__(self, parent, opts: dict, roi_box: tuple[int, int, int, int] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export map")
        self._opts, self._roi_box = opts, roi_box
        ny, nx = opts["height_px"], opts["width_px"]
        lay = QVBoxLayout(self)
        upp = opts.get("um_per_px")
        size = f"Map: {nx} × {ny} px" + (f"  ({nx * upp:.1f} × {ny * upp:.1f} µm)" if upp else "")
        lay.addWidget(QLabel(size))

        files = QGroupBox("Files")
        fl = QVBoxLayout(files)
        self.chk_tif = QCheckBox("Values: 32-bit TIFF (ImageJ, Igor, analysis)")
        self.chk_png = QCheckBox("Picture: PNG")
        self.chk_pdf = QCheckBox("Picture: PDF (vector, editable text in Illustrator)")
        self.chk_svg = QCheckBox("Picture: SVG")
        for c, on in ((self.chk_tif, True), (self.chk_png, True), (self.chk_pdf, False), (self.chk_svg, False)):
            c.setChecked(on)
            fl.addWidget(c)
        lay.addWidget(files)

        reg = QGroupBox("Region of the map")
        rl = QVBoxLayout(reg)
        self.rb_all = QRadioButton(f"Whole map ({nx} × {ny} px)")
        self.rb_roi = QRadioButton("Map ROI" + (f" (x {roi_box[0]}–{roi_box[1]}, y {roi_box[2]}–{roi_box[3]})"
                                                if roi_box else " (none drawn)"))
        self.rb_roi.setEnabled(roi_box is not None)
        self.rb_typed = QRadioButton("Typed: x0, x1, y0, y1 (px)")
        self.crop_edit = QLineEdit(f"0, {nx}, 0, {ny}")
        self.rb_all.setChecked(True)
        for w in (self.rb_all, self.rb_roi, self.rb_typed, self.crop_edit):
            rl.addWidget(w)
        for rb in (self.rb_all, self.rb_roi, self.rb_typed):
            rb.toggled.connect(self._region_changed)
        self.crop_edit.editingFinished.connect(self._region_changed)
        lay.addWidget(reg)

        pic = QGroupBox("Picture size")
        pf = QFormLayout(pic)
        row = QHBoxLayout()
        self.w_spin, self.h_spin = _ispin(16, 20000, nx), _ispin(16, 20000, ny)
        row.addWidget(QLabel("width"))
        row.addWidget(self.w_spin)
        row.addWidget(QLabel("× height"))
        row.addWidget(self.h_spin)
        row.addWidget(QLabel("px"))
        pf.addRow(row)
        self.chk_aspect = QCheckBox("Keep the map's aspect ratio")
        self.chk_aspect.setChecked(True)
        pf.addRow(self.chk_aspect)
        self.dpi_spin = _ispin(30, 1200, 300)
        pf.addRow("Resolution (dpi):", self.dpi_spin)
        self.inch_label = QLabel()
        pf.addRow("Printed size:", self.inch_label)
        self.chk_auto_size = QCheckBox("Automatic size instead (6 × 5.2 in, margins trimmed)")
        pf.addRow(self.chk_auto_size)
        self.w_spin.valueChanged.connect(lambda v: self._keep_aspect("w"))
        self.h_spin.valueChanged.connect(lambda v: self._keep_aspect("h"))
        self.dpi_spin.valueChanged.connect(self._update_inches)
        lay.addWidget(pic)

        look = QGroupBox("Look")
        lf = QFormLayout(look)
        self.cmap = QComboBox()
        self.cmap.addItems(list(dict.fromkeys((opts["cmap"],) + CMAPS)))
        lf.addRow("Colour map:", self.cmap)
        lim = QHBoxLayout()
        self.vmin = _dspin(-1e12, 1e12, float(opts["vmin"]), max((opts["vmax"] - opts["vmin"]) / 50, 1e-6), 4)
        self.vmax = _dspin(-1e12, 1e12, float(opts["vmax"]), max((opts["vmax"] - opts["vmin"]) / 50, 1e-6), 4)
        lim.addWidget(self.vmin)
        lim.addWidget(QLabel("to"))
        lim.addWidget(self.vmax)
        b = QPushButton("As shown")
        b.setToolTip("Back to the range the viewer shows now")
        b.clicked.connect(lambda: (self.vmin.setValue(float(opts["vmin"])), self.vmax.setValue(float(opts["vmax"]))))
        lim.addWidget(b)
        lf.addRow("Colour range:", lim)
        self.chk_cbar = QCheckBox("Colour bar")
        self.chk_cbar.setChecked(True)
        lf.addRow(self.chk_cbar)
        self.label_edit = QLineEdit(opts.get("label", ""))
        lf.addRow("Colour-bar label:", self.label_edit)
        self.title_edit = QLineEdit(opts.get("title", ""))
        lf.addRow("Title:", self.title_edit)
        self.chk_scalebar = QCheckBox("Scale bar (µm)")
        self.chk_scalebar.setEnabled(bool(upp))
        lf.addRow(self.chk_scalebar)
        self.chk_image_only = QCheckBox("Image only: exactly width × height px, no colour bar or title")
        lf.addRow(self.chk_image_only)
        lay.addWidget(look)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.error = QLabel()
        self.error.setStyleSheet("color: #c0392b")
        lay.addWidget(self.error)
        self._busy = False
        self._region_changed()

    # ------------------------------------------------------------------ state
    def crop(self) -> tuple[int, int, int, int] | None:
        if self.rb_roi.isChecked() and self._roi_box:
            return tuple(int(v) for v in self._roi_box)
        if self.rb_typed.isChecked():
            return tuple(int(round(v)) for v in _numbers(self.crop_edit.text(), 4))
        return None

    def _region_size(self) -> tuple[int, int]:
        try:
            c = self.crop()
        except ValueError:
            c = None
        if c is None:
            return self._opts["width_px"], self._opts["height_px"]
        return max(1, abs(c[1] - c[0])), max(1, abs(c[3] - c[2]))

    def _region_changed(self, *_):
        self.crop_edit.setEnabled(self.rb_typed.isChecked())
        w, h = self._region_size()
        self._busy = True
        self.w_spin.setValue(w)
        self.h_spin.setValue(h)
        self._busy = False
        self._update_inches()

    def _keep_aspect(self, which: str):
        if not self._busy and self.chk_aspect.isChecked():
            w, h = self._region_size()
            self._busy = True
            if which == "w":
                self.h_spin.setValue(max(16, round(self.w_spin.value() * h / w)))
            else:
                self.w_spin.setValue(max(16, round(self.h_spin.value() * w / h)))
            self._busy = False
        self._update_inches()

    def _update_inches(self, *_):
        d = self.dpi_spin.value()
        w, h = self.w_spin.value() / d, self.h_spin.value() / d
        self.inch_label.setText(f"{w:.2f} × {h:.2f} in  ({w * 2.54:.1f} × {h * 2.54:.1f} cm)")

    def options(self) -> dict:
        """Keyword arguments for PeemService.export_map (without name / kind / path / var)."""
        pictures = tuple(f for f, c in (("png", self.chk_png), ("pdf", self.chk_pdf), ("svg", self.chk_svg))
                         if c.isChecked())
        return {"values": self.chk_tif.isChecked(), "pictures": pictures, "crop": self.crop(),
                "size_px": None if self.chk_auto_size.isChecked() else (self.w_spin.value(), self.h_spin.value()),
                "dpi": self.dpi_spin.value(), "cmap": self.cmap.currentText(),
                "vmin": self.vmin.value(), "vmax": self.vmax.value(),
                "colorbar": self.chk_cbar.isChecked(), "title": self.title_edit.text(),
                "label": self.label_edit.text(), "scalebar": self.chk_scalebar.isChecked(),
                "image_only": self.chk_image_only.isChecked()}

    def _accept(self):
        try:
            o = self.options()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        if not o["values"] and not o["pictures"]:
            self.error.setText("Tick at least one file.")
            return
        if o["vmax"] <= o["vmin"]:
            self.error.setText("The colour maximum must be above the minimum.")
            return
        self.accept()
