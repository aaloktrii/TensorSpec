"""Play / pause a frame slider as a movie (reusable by any panel with a frame QSlider)."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox, QWidget


class StackPlayer(QWidget):
    """▶ / ⏸ button + frames-per-second + loop, driving an existing frame slider.

    The slider keeps its own valueChanged connection, so the panel redraws each
    frame exactly as when the slider is dragged by hand.
    """

    def __init__(self, slider: QSlider, fps: int = 5, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._step)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.btn = QPushButton("▶ Play")
        self.btn.setFixedWidth(80)
        self.btn.clicked.connect(self.toggle)
        lay.addWidget(self.btn)
        self.fps = QSpinBox()
        self.fps.setRange(1, 30)
        self.fps.setValue(fps)
        self.fps.setSuffix(" fps")
        self.fps.valueChanged.connect(self._set_interval)
        lay.addWidget(self.fps)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        lay.addWidget(self.loop)
        self.label = QLabel("")
        lay.addWidget(self.label)
        lay.addStretch(1)
        slider.valueChanged.connect(self._update_label)
        slider.rangeChanged.connect(lambda *_: self._update_label())
        self._update_label()

    @property
    def playing(self) -> bool:
        return self.timer.isActive()

    def toggle(self):
        if self.playing:
            self.stop()
        elif self.slider.isEnabled() and self.slider.maximum() > self.slider.minimum():
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(self.slider.minimum())
            self._set_interval()
            self.timer.start()
            self.btn.setText("⏸ Pause")

    def stop(self):
        self.timer.stop()
        self.btn.setText("▶ Play")

    def _set_interval(self, *_):
        self.timer.setInterval(int(1000 / max(1, self.fps.value())))

    def _step(self):
        if not self.slider.isEnabled():
            self.stop()
            return
        v = self.slider.value() + 1
        if v > self.slider.maximum():
            if not self.loop.isChecked():
                self.stop()
                return
            v = self.slider.minimum()
        self.slider.setValue(v)

    def _update_label(self, *_):
        n = self.slider.maximum() - self.slider.minimum() + 1
        self.label.setText(f"frame {self.slider.value() + 1} / {n}" if n > 1 else "")
