import sys
from PyQt5 import QtCore, QtWidgets

from qt_centerdetector import QtCenterDetector
from qt_stackprocessor import QtStackProcessor


class PlaceholderPanel(QtWidgets.QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(f"{title} is not migrated yet.")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)


class TwoDTwoPQt(QtWidgets.QMainWindow):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.setWindowTitle("2D2P Automated Z-Drift Corrector (Qt)")

        central = QtWidgets.QWidget()
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)

        top_bar = QtWidgets.QHBoxLayout()
        self.set_folder_btn = QtWidgets.QPushButton("Set Data Folder")
        self.set_folder_btn.setStyleSheet(
            "QPushButton {"
            "background-color: #2563eb;"
            "color: white;"
            "font-weight: 600;"
            "padding: 6px 12px;"
            "border-radius: 4px;"
            "}"
            "QPushButton:hover { background-color: #1d4ed8; }"
            "QPushButton:pressed { background-color: #1e40af; }"
        )
        self.set_folder_btn.clicked.connect(self.select_data_folder)
        self.folder_label = QtWidgets.QLabel(self.folder)
        self.folder_label.setStyleSheet("QLabel { color: #334155; }")
        top_bar.addWidget(self.set_folder_btn)
        top_bar.addWidget(self.folder_label, 1)

        self.tabs = QtWidgets.QTabWidget()

        central_layout.addLayout(top_bar)
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QWidget { font-family: Arial, 'Segoe UI', sans-serif; }"
            "QPushButton { font-size: 11pt; padding: 6px 10px; }"
            "QLabel { font-size: 10.5pt; }"
            "QLineEdit { font-size: 10.5pt; padding: 4px; }"
            "QTabBar::tab {"
            "font-size: 12pt;"
            "padding: 8px 14px;"
            "min-width: 170px;"
            "border: 1px solid #cbd5f5;"
            "}"
            "QTabBar::tab:first { background: #93c5fd; }"
            "QTabBar::tab:middle { background: #86efac; }"
            "QTabBar::tab:last { background: #fde68a; }"
            "QTabBar::tab:selected {"
            "border: 1px solid #94a3b8;"
            "}"
        )

        self.center_panel = QtCenterDetector(folder=self.folder, app=self)
        self.tabs.addTab(self.center_panel, "Center Detector")

        self.stack_panel = QtStackProcessor(folder=self.folder, app=self)
        self.tabs.addTab(self.stack_panel, "Stack Processor")
        self.tabs.addTab(PlaceholderPanel("Z-Drift Monitor"), "Z-Drift Monitor")

        self._build_log_dock()

    def _build_log_dock(self):
        dock = QtWidgets.QDockWidget("Log Output", self)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea
            | QtCore.Qt.RightDockWidgetArea
            | QtCore.Qt.BottomDockWidgetArea
        )
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QPlainTextEdit {"
            "background-color: #0f172a;"
            "color: #e2e8f0;"
            "border: 1px solid #1e293b;"
            "border-radius: 6px;"
            "font-family: Arial, 'Segoe UI', sans-serif;"
            "font-size: 11pt;"
            "padding: 6px;"
            "}"
        )
        dock.setWidget(self.log_text)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def log_message(self, message):
        self.log_text.appendPlainText(message)
        self.log_text.appendPlainText("-" * 20)

    def select_data_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Data Folder", self.folder or ""
        )
        if not path:
            return
        self.folder = path
        self.folder_label.setText(path)
        if hasattr(self.center_panel, "set_folder"):
            self.center_panel.set_folder(path)
        if hasattr(self.stack_panel, "set_folder"):
            self.stack_panel.set_folder(path)


def main():
    app = QtWidgets.QApplication(sys.argv)
    folder_path = "D:/data/"
    window = TwoDTwoPQt(folder_path)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
