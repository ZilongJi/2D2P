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

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.setStyleSheet(
            "QWidget { font-family: Arial, 'Segoe UI', sans-serif; }"
            "QPushButton { font-size: 11pt; padding: 6px 10px; }"
            "QLabel { font-size: 10.5pt; }"
            "QLineEdit { font-size: 10.5pt; padding: 4px; }"
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    folder_path = "C:/Users/BurgessLab/Desktop/SM2PCode/2D2P/Data"
    window = TwoDTwoPQt(folder_path)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
