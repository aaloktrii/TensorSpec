"""Layout that places widgets left to right and wraps to a new line when the panel is too narrow
(Qt's flow-layout example), so control rows never squeeze widgets on top of each other."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QHBoxLayout, QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    def __init__(self, parent=None, h_spacing: int = 10, v_spacing: int = 4):
        super().__init__(parent)
        self._items = []
        self._h, self._v = h_spacing, v_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        need = self._place(rect, apply=True)
        # When a control grows (e.g. the Sym ± 'white at …' label appears) the row can wrap onto another
        # line; the parent layout keeps its cached height, so the new line would be cut off. Asking for
        # the needed height makes the parent give it.
        w = self.parentWidget()
        if w is not None and w.minimumHeight() != need:
            w.setMinimumHeight(need)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect, apply: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for item in self._items:
            if item.widget() is not None and not item.widget().isVisibleTo(item.widget().parentWidget()):
                continue
            hint = item.sizeHint()
            if x > r.x() and x + hint.width() > r.right() + 1:
                x, y, line_h = r.x(), y + line_h + self._v, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def group(*widgets, spacing: int = 4) -> QWidget:
    """Widgets that stay together on one line of a FlowLayout (e.g. a label and its control)."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        lay.addWidget(w)
    box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return box
