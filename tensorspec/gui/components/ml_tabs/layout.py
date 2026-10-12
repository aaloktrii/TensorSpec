"""Layout helpers shared by the ML suite panels.

Every panel puts its controls in a scroll area above its canvas, split by a
drag handle, so the canvas keeps its height when the controls are tall.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget


def scrollable(widget):
    """Wrap a control column so it can shrink below its natural height."""
    scroll = QScrollArea()
    scroll.setWidget(widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    return scroll


def split_panel(controls, canvas, sizes=(300, 460)):
    """Scrollable controls above a canvas, divided by a drag handle.

    The canvas gets the stretch so it absorbs any extra height, and the
    controls stay reachable via the scroll area when the pane is short.
    """
    # Inset the canvas to line up with the control column, which the scroll
    # area indents by its own layout margin.
    canvas_holder = QWidget()
    holder_layout = QVBoxLayout(canvas_holder)
    holder_layout.setContentsMargins(9, 0, 9, 9)
    holder_layout.addWidget(canvas)

    splitter = QSplitter(Qt.Orientation.Vertical)
    splitter.addWidget(scrollable(controls))
    splitter.addWidget(canvas_holder)
    splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes(list(sizes))

    page = QWidget()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(0, 0, 0, 0)
    page_layout.addWidget(splitter)
    return page


def tab_group(pages):
    """Build one top-level group's page.

    A group holding a single tab renders that tab directly, so we don't draw
    a nested bar with only one entry in it.
    """
    if len(pages) == 1:
        return pages[0][1]

    inner = QTabWidget()
    inner.setDocumentMode(True)
    inner.setUsesScrollButtons(True)
    inner.setElideMode(Qt.TextElideMode.ElideNone)
    for label, page in pages:
        inner.addTab(page, label)
    return inner
