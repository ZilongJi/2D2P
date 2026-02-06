import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
import pyqtgraph as pg

import tifffile
from utils_image import get_meanZstack
from utils_io import get_scanimage_frame_times


class _TiffStackAdapter:
    def __init__(self, frames, angles):
        self._frames = frames
        self._angles = np.asarray(angles, dtype=float)

    def get_frame(self, index):
        # 1-based index to match SITiffIO behavior.
        return self._frames[index - 1]

    def get_all_theta(self):
        return self._angles

    def get_n_frames(self):
        return int(self._frames.shape[0])


def _read_relog_angles(relog_path, n_frames):
    data = np.genfromtxt(relog_path, dtype=float, comments="#", invalid_raise=False)
    if data.size == 0:
        raise ValueError("RElog file appears empty or unreadable.")
    if data.ndim == 1:
        angles = data
    else:
        angles = data[:, -1]

    angles = np.asarray(angles, dtype=float)
    angles = angles[~np.isnan(angles)]
    if angles.size == 0:
        raise ValueError("No numeric angle data found in RElog file.")

    if angles.size != n_frames:
        x_src = np.linspace(0.0, 1.0, angles.size)
        x_tgt = np.linspace(0.0, 1.0, n_frames)
        angles = np.interp(x_tgt, x_src, angles)

    return angles


class QtStackProcessor(QtWidgets.QWidget):
    def __init__(self, folder=None, app=None, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.app = app

        self.tifffilename = None
        self.relogfilename = None
        self.CentreDetectionFolder = None
        self.meanStacks = None
        self.display_index = 0

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)

        self.import_tiff_btn = QtWidgets.QPushButton("Import ZStack")
        self.import_tiff_btn.clicked.connect(self.import_tiff)
        layout.addWidget(self.import_tiff_btn, 0, 0)

        self.import_relog_btn = QtWidgets.QPushButton("Import RElog file")
        self.import_relog_btn.clicked.connect(self.import_RElog)
        layout.addWidget(self.import_relog_btn, 1, 0)

        fields_row = QtWidgets.QHBoxLayout()
        fields_row.setSpacing(8)

        fields_row.addWidget(QtWidgets.QLabel("Volumes:"))
        self.volumes = QtWidgets.QLineEdit("1")
        self.volumes.setFixedWidth(60)
        fields_row.addWidget(self.volumes)

        fields_row.addWidget(QtWidgets.QLabel("Stacks:"))
        self.stacks = QtWidgets.QLineEdit("41")
        self.stacks.setFixedWidth(60)
        fields_row.addWidget(self.stacks)

        fields_row.addWidget(QtWidgets.QLabel("Frames:"))
        self.frames = QtWidgets.QLineEdit("200")
        self.frames.setFixedWidth(60)
        fields_row.addWidget(self.frames)

        self.process_btn = QtWidgets.QPushButton("Get mean Zstacks")
        self.process_btn.clicked.connect(self.getmeanzstack)
        fields_row.addWidget(self.process_btn)
        fields_row.addStretch(1)
        layout.addLayout(fields_row, 2, 0, 1, 3)

        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.prev_btn.clicked.connect(self.previous_image)
        layout.addWidget(self.prev_btn, 3, 0)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)
        layout.addWidget(self.next_btn, 3, 1, 1, 2)

        self.index_label = QtWidgets.QLabel("Stack index: -")
        layout.addWidget(self.index_label, 4, 0, 1, 3)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        layout.addWidget(self.status_label, 5, 0, 1, 3)

        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        layout.addWidget(self.image_view, 6, 0, 1, 3)

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

        self.CentreDetectionFolder = os.path.join(
            os.path.dirname(self.tifffilename), "CentreDetectionResults"
        )

    def import_RElog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", self.folder or "", "txt files (*.txt);;all files (*.*)"
        )
        if not filename:
            return

        self.relogfilename = filename
        self.log_message(f"Imported RElog file: {self.relogfilename}")

    def getmeanzstack(self):
        if not self.tifffilename or not self.relogfilename:
            self.log_message("Error: Please import both a tiff file and a RElog file.")
            return
        if self.CentreDetectionFolder is None:
            self.log_message(
                "Error: CentreDetectionResults folder not set. Run center detection first."
            )
            return

        self.status_label.setText("Status: Loading...")
        QtWidgets.QApplication.processEvents()

        circlecenterfilename = os.path.join(
            self.CentreDetectionFolder, "circlecenter.txt"
        )
        if not os.path.exists(circlecenterfilename):
            self.log_message(
                "Error: circlecenter.txt not found in CentreDetectionResults folder."
            )
            self.status_label.setText("Status: Idle")
            return
        with open(circlecenterfilename, "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
            self.rotx = float(last_line.split()[0])
            self.roty = float(last_line.split()[1])

        try:
            num_v = int(self.volumes.text())
            num_s = int(self.stacks.text())
            num_f = int(self.frames.text())
        except ValueError:
            self.log_message("Error! Enter the number of volumes, stacks and frames.")
            self.status_label.setText("Status: Idle")
            return

        self.log_message("Reading tiff file with tifffile...")
        frames = tifffile.imread(self.tifffilename)
        if frames.ndim != 3:
            self.log_message("Error: Expected a 3D tiff stack [n_frames, height, width].")
            self.status_label.setText("Status: Idle")
            return

        n_frames = frames.shape[0]
        expected = num_v * num_s * num_f
        if expected != n_frames:
            self.log_message(
                f"Warning: volumes*stacks*frames = {expected}, but tiff has {n_frames} frames."
            )

        self.log_message("Reading RElog angles...")
        try:
            angles = _read_relog_angles(self.relogfilename, n_frames)
        except ValueError as exc:
            self.log_message(f"Error: {exc}")
            self.status_label.setText("Status: Idle")
            return

        S = _TiffStackAdapter(frames, angles)
        self.log_message("Get mean Zstacks by unrotating and cropping each frame, then averaging each stack.")
        self.meanStacks = get_meanZstack(
            S, num_v, num_s, num_f, Rotcenter=[self.rotx, self.roty], ImgReg=True
        )

        self.log_message(
            "Save the mean Zstacks as npy files and png images to the CentreDetectionResults folder."
        )
        np.save(
            os.path.join(self.CentreDetectionFolder, "meanstacks.npy"),
            self.meanStacks,
        )

        zstack_folder = os.path.join(self.CentreDetectionFolder, "zstacks")
        if os.path.exists(zstack_folder):
            shutil.rmtree(zstack_folder)
        os.mkdir(zstack_folder)

        for i in range(self.meanStacks.shape[0]):
            fig = plt.figure()
            plt.imshow(self.meanStacks[i, :, :], cmap="gray")
            plt.axis("off")
            plt.savefig(
                os.path.join(zstack_folder, f"stack{i + 1}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)

        self.display_index = 0
        self.display_processed_images()
        self.log_message("Unrotation and crop finished.")
        self.status_label.setText("Status: Done")

    def display_processed_images(self):
        if self.meanStacks is None or self.meanStacks.size == 0:
            return

        image = self.meanStacks[self.display_index].astype(np.float32)
        denom = max(image.max() - image.min(), 1e-6)
        image = (image - image.min()) / denom

        self.image_view.setImage(image.T, autoLevels=True)
        self.index_label.setText(f"Stack index: {self.display_index + 1}/{self.meanStacks.shape[0]}")

    def next_image(self):
        if self.meanStacks is None:
            return
        if self.display_index < len(self.meanStacks) - 1:
            self.display_index += 1
        self.display_processed_images()

    def previous_image(self):
        if self.meanStacks is None:
            return
        if self.display_index > 0:
            self.display_index -= 1
        self.display_processed_images()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = QtStackProcessor(folder=os.getcwd())
    widget.show()
    app.exec_()
