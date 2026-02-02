import os
import shutil
import numpy as np

from PyQt5 import QtWidgets
import pyqtgraph as pg

try:
    from scanimagetiffio import SITiffIO
except Exception:
    SITiffIO = None

from utils_image import getMeanTiff_randomsampling, getMeanTiff_equalsampling


class QtCenterDetector(QtWidgets.QWidget):
    def __init__(self, folder=None, app=None, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.app = app

        self.tifffilename = None
        self.relogfilename = None
        self.DPfolder = None
        self.circlecenterfile = None
        self.spaceclicks = 0
        self.circlecenter = np.asarray([0.0, 0.0])
        self.roi = None

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)

        self.import_tiff_btn = QtWidgets.QPushButton("Import tiff")
        self.import_tiff_btn.clicked.connect(self.import_tiff)
        layout.addWidget(self.import_tiff_btn, 0, 0)

        layout.addWidget(QtWidgets.QLabel("Ave. Frac/Bins"), 1, 0)
        self.fraction_or_bins = QtWidgets.QLineEdit()
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

        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, 4, 0, 1, 2)

        self.angle_plot = pg.PlotWidget()
        self.angle_plot.setBackground("w")
        self.angle_plot.setLabel("left", "Count")
        self.angle_plot.setLabel("bottom", "Angle (deg)")
        layout.addWidget(self.angle_plot, 5, 0, 1, 2)

        self.status_label = QtWidgets.QLabel(
            "Tip: Click 'Add/Reset Circle ROI' then drag/resize the circle."
        )
        layout.addWidget(self.status_label, 6, 0, 1, 2)

    def log_message(self, message):
        if self.app is not None:
            self.app.log_message(message)

    def import_tiff(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", self.folder or "", "tiff files (*.tif);;all files (*.*)"
        )
        if not filename:
            return

        self.tifffilename = filename
        self.log_message(f"Imported tiff file: {self.tifffilename}")

        self.DPfolder = os.path.join(os.path.dirname(self.tifffilename), "DP")
        if os.path.exists(self.DPfolder):
            shutil.rmtree(self.DPfolder)
        os.mkdir(self.DPfolder)

        self.circlecenterfile = os.path.join(self.DPfolder, "circlecenter.txt")
        with open(self.circlecenterfile, "w", encoding="utf-8") as f:
            f.write("")

    def import_RElog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", self.folder or "", "txt files (*.txt);;all files (*.*)"
        )
        if not filename:
            return
        self.relogfilename = filename
        self.log_message(f"Imported RElog file: {self.relogfilename}")

    def averagetif(self):
        if SITiffIO is None:
            self.log_message("Error: scanimagetiffio.SITiffIO not available.")
            return
        if not self.tifffilename or not self.relogfilename:
            self.log_message("Error: Please import both a tiff and a RElog file.")
            return

        self.log_message("Reading tiff file...")
        S = SITiffIO()
        S.open_tiff_file(self.tifffilename, "r")
        S.open_rotary_file(self.relogfilename)
        S.interp_times()
        self.S = S
        self.log_message(f"Counted {S.get_n_frames()} frames in the tif file.")
        self.log_message("Done reading tiff file.")

        try:
            frac_or_bins = float(self.fraction_or_bins.text())
        except ValueError:
            self.log_message("Error! Please enter a valid fraction number.")
            return

        if frac_or_bins < 1:
            self.log_message("Get the averaged image by random sampling...")
            self.meantif = getMeanTiff_randomsampling(self.S, frac=frac_or_bins)
        else:
            self.log_message("Get the averaged image by equal sampling...")
            self.meantif = getMeanTiff_equalsampling(self.S, numBins=int(frac_or_bins))

        self.display_image(self.meantif)

        angles = self.S.get_all_theta()
        self.display_angles(angles)

    def display_angles(self, angles):
        if angles is None:
            return
        angles = np.asarray(angles)
        self.angle_plot.clear()
        y, x = np.histogram(angles, bins=50)
        self.angle_plot.plot(x[:-1], y, stepMode=True, fillLevel=0, brush=(150, 150, 150, 120))

    def display_image(self, image):
        if image is None:
            return
        img = image.astype(np.float32)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        self.image_view.setImage(img.T, autoLevels=True)

        if self.DPfolder:
            out_path = os.path.join(self.DPfolder, "averagedTiff.npy")
            np.save(out_path, image)

    def add_or_reset_roi(self):
        if self.meantif is None:
            self.log_message("Please calculate the mean frame first.")
            return

        if self.roi is not None:
            self.image_view.removeItem(self.roi)
            self.roi = None

        h, w = self.meantif.shape[:2]
        r = min(h, w) * 0.1
        self.roi = pg.CircleROI(
            [w / 2 - r, h / 2 - r],
            [2 * r, 2 * r],
            pen=pg.mkPen("r", width=2),
        )
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

        self.spaceclicks += 1
        self.circlecenter = (1 - 1 / self.spaceclicks) * self.circlecenter + (
            1 / self.spaceclicks
        ) * np.array([x, y])
        self.circlecenter = np.round(self.circlecenter, 0)

        self.log_message(f"Center position: {x:.1f} {y:.1f}")
        self.log_message(f"Number of saves: {self.spaceclicks}")
        self.log_message(f"Updated center position: {self.circlecenter}")

        with open(self.circlecenterfile, "a", encoding="utf-8") as f:
            f.write(f"{self.circlecenter[0]} {self.circlecenter[1]}\n")
