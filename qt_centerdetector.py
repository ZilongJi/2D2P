import os
import shutil
import numpy as np

from PyQt5 import QtWidgets
import pyqtgraph as pg

import tifffile
from utils_image import getMeantiff


class QtCenterDetector(QtWidgets.QWidget):
    def __init__(self, folder=None, app=None, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.app = app

        self.tifffilename = None
        self.savefolder = None
        self.circlecenterfile = None
        self.center_save_count = 0
        self.circlecenter = None
        self.roi = None

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)

        self.import_tiff_btn = QtWidgets.QPushButton("Import tiff")
        self.import_tiff_btn.clicked.connect(self.import_tiff)
        layout.addWidget(self.import_tiff_btn, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Fraction of frames to get the mean"), 1, 0)
        self.fraction_or_bins = QtWidgets.QLineEdit("1.0")
        layout.addWidget(self.fraction_or_bins, 2, 0)

        self.calc_mean_btn = QtWidgets.QPushButton("Calculate mean frame")
        self.calc_mean_btn.clicked.connect(self.averagetif)
        layout.addWidget(self.calc_mean_btn, 0, 1, 3, 1)

        self.add_roi_btn = QtWidgets.QPushButton("Add/Reset Circle ROI")
        self.add_roi_btn.clicked.connect(self.add_or_reset_roi)
        layout.addWidget(self.add_roi_btn, 3, 0, 1, 1)

        self.save_center_btn = QtWidgets.QPushButton("Save Center")
        self.save_center_btn.clicked.connect(self.save_center_position)
        layout.addWidget(self.save_center_btn, 3, 1, 1, 1)

        self.progress_label = QtWidgets.QLabel("Status: Idle")
        layout.addWidget(self.progress_label, 4, 0, 1, 2)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, 5, 0, 1, 2)

        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, 6, 0, 1, 2)

        self.status_label = QtWidgets.QLabel(
            "Tip: Click 'Add/Reset Circle ROI' then drag/resize the circle."
        )
        layout.addWidget(self.status_label, 7, 0, 1, 2)

    def log_message(self, message):
        if self.app is not None:
            self.app.log_message(message)
        
    def set_folder(self, folder):
        self.folder = folder

    def import_tiff(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", self.folder or "", "tiff files (*.tif);;all files (*.*)"
        )
        if not filename:
            return

        self.tifffilename = filename
        self.log_message(f"Imported tiff file: {self.tifffilename}")

        self.savefolder = os.path.join(os.path.dirname(self.tifffilename), "CentreDetectionResults")
        if os.path.exists(self.savefolder):
            shutil.rmtree(self.savefolder)
        os.mkdir(self.savefolder)

        self.circlecenterfile = os.path.join(self.savefolder, "circlecenter.txt")
        with open(self.circlecenterfile, "w", encoding="utf-8") as f:
            f.write("")

    def averagetif(self):
        if not self.tifffilename:
            self.log_message("Error: Please import a tiff before detecting the rotating centre!")
            return

        self.progress_label.setText("Status: Loading TIFF and computing mean...")
        self.progress_bar.setRange(0, 0)
        QtWidgets.QApplication.processEvents()

        self.log_message("Reading tiff file with python tifffile package...")
        tiffdata = tifffile.imread(self.tifffilename)
        if tiffdata.ndim != 3:
            self.log_message("Error: Expected a 3D tiff stack [n_frames, height, width].")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Status: Idle")
            return
        self.log_message(f"Counted {tiffdata.shape[0]} frames in the tif file.")
        self.log_message("Done reading tiff file.")

        try:
            frac_or_bins = float(self.fraction_or_bins.text())
        except ValueError:
            self.log_message("Error! Please enter a valid fraction number.")
            return

        self.log_message("Get the averaged image by random sampling a fraction...")
        self.meantif = getMeantiff(tiffdata, frac=frac_or_bins)

        self.display_image(self.meantif)

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_label.setText("Status: Done")

    def display_image(self, image):
        if image is None:
            return
        img = image.astype(np.float32)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        self.image_view.setImage(img.T, autoLevels=True)

        if self.savefolder:
            out_path = os.path.join(self.savefolder, "averagedTiff.npy")
            np.save(out_path, image)

    def add_or_reset_roi(self):
        if self.meantif is None:
            self.log_message("Please calculate the mean frame first.")
            return

        if self.roi is not None:
            self.image_view.removeItem(self.roi)
            self.roi = None
            self.center_save_count = 0
            self.circlecenter = None
            self.log_message("Reset saved center running average.")
            if self.circlecenterfile and os.path.exists(self.circlecenterfile):
                with open(self.circlecenterfile, "w", encoding="utf-8") as f:
                    f.write("")
                self.log_message("Cleared circlecenter.txt.")

        h, w = self.meantif.shape[:2]
        r = min(h, w) * 0.1
        self.roi = pg.CircleROI(
            [w / 2 - r, h / 2 - r],
            [2 * r, 2 * r],
            pen=pg.mkPen("r", width=2),
        )
        if hasattr(self.roi, "setHandleSize"):
            self.roi.setHandleSize(20)
        if hasattr(self.roi, "setHandlePen"):
            self.roi.setHandlePen(pg.mkPen("g", width=2))
        if hasattr(self.roi, "setHandleHoverPen"):
            self.roi.setHandleHoverPen(pg.mkPen("g", width=2))
        if hasattr(self.roi, "setHandleBrush"):
            self.roi.setHandleBrush(pg.mkBrush(0, 255, 0))
        self.image_view.addItem(self.roi)

    def save_center_position(self):
        if self.roi is None:
            self.log_message("Please add a circle ROI first.")
            return
        if not self.circlecenterfile:
            self.log_message("Please import a tiff file first.")
            return

        pos = self.roi.pos()
        size = self.roi.size()
        x = float(pos.x() + size.x() / 2.0)
        y = float(pos.y() + size.y() / 2.0)

        if self.center_save_count == 0 or self.circlecenter is None:
            self.center_save_count = 1
            self.circlecenter = np.array([x, y], dtype=float)
        else:
            self.center_save_count += 1
            self.circlecenter = (1 - 1 / self.center_save_count) * self.circlecenter + (
                1 / self.center_save_count
            ) * np.array([x, y])
        self.circlecenter = np.round(self.circlecenter, 0)

        self.log_message(f"Center position: {x:.1f} {y:.1f}")
        self.log_message(f"Number of saves: {self.center_save_count}")
        self.log_message(f"Updated center position: {self.circlecenter}")

        with open(self.circlecenterfile, "a", encoding="utf-8") as f:
            f.write(f"{self.circlecenter[0]} {self.circlecenter[1]}\n")
