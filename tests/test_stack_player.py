"""StackPlayer drives a frame slider: play, loop, stop at the end without loop."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QSlider  # noqa: E402

from tensorspec.gui.components.stack_player import StackPlayer  # noqa: E402


def test_play_loop_and_stop():
    app = QApplication.instance() or QApplication([])
    s = QSlider(Qt.Horizontal)
    s.setRange(0, 2)
    p = StackPlayer(s)
    assert p.label.text() == "frame 1 / 3"
    p.toggle()
    assert p.playing and p.btn.text() == "⏸ Pause"
    for want in (1, 2, 0):  # loops back to the first frame
        p._step()
        assert s.value() == want
    p.loop.setChecked(False)
    s.setValue(2)
    p._step()
    assert not p.playing and s.value() == 2
    s.setEnabled(False)
    p.toggle()
    assert not p.playing  # nothing to play when the slider is disabled
    app.processEvents()
