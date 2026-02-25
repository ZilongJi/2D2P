import os
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from PyQt6 import QtCore, QtGui, QtWidgets

import torch
import tifffile
from suite2p import default_ops
from suite2p.registration import register

from utils_image import get_unrotate_crop_cv2, RegFrame, compute_zpos_sp, findFOV
from utils_io import get_frame_angles_from_rotary


class QtZDriftProcessor(QtWidgets.QWidget):
    def __init__(self, folder=None, app=None, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.app = app

        self.tifffilename = None
        self.relogfilename = None
        self.DPFolder = None

        self.meanRegImg = None
        self.regFrames = None
        self.corrMatrix = None

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)

        self.import_tiff_btn = QtWidgets.QPushButton("Import tiff")
        self.import_tiff_btn.clicked.connect(self.import_tiff)
        layout.addWidget(self.import_tiff_btn, 0, 0)

        self.import_relog_btn = QtWidgets.QPushButton("Import RElog")
        self.import_relog_btn.clicked.connect(self.import_RElog)
        layout.addWidget(self.import_relog_btn, 1, 0)

        self.corr_btn = QtWidgets.QPushButton("Correlation Analysis")
        self.corr_btn.clicked.connect(self.correlationanalysis)
        layout.addWidget(self.corr_btn, 0, 1, 2, 1)

        self.reg_image = QtWidgets.QLabel()
        self.reg_image.setFixedSize(512, 512)
        self.reg_image.setStyleSheet("background-color: #4d4d4d;")
        self.reg_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.reg_image, 4, 0, 1, 2)

        self.corr_image = QtWidgets.QLabel()
        self.corr_image.setFixedSize(512, 256)
        self.corr_image.setStyleSheet("background-color: #ffffff;")
        self.corr_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.corr_image, 5, 0, 1, 2)

    def log_message(self, message):
        if self.app is not None:
            self.app.log_message(message)

    def set_folder(self, folder):
        self.folder = folder

    def import_tiff(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a tiff file", self.folder or "", "tiff files (*.tif);;all files (*.*)"
        )
        if not filename:
            return
        self.tifffilename = filename
        self.log_message(f"Imported tiff file: {self.tifffilename}")

        self.DPFolder = os.path.join(os.path.dirname(self.tifffilename), "DP")
        os.makedirs(self.DPFolder, exist_ok=True)

    def import_RElog(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a RElog file", self.folder or "", "txt files (*.txt);;all files (*.*)"
        )
        if not filename:
            return
        self.relogfilename = filename
        self.log_message(f"Imported RElog file: {self.relogfilename}")

    def correlationanalysis(self):
        if not self.tifffilename or not self.relogfilename:
            self.log_message("Error: Please import both tiff and RElog files.")
            return
        if not self.DPFolder:
            self.log_message("Error: DP folder not set.")
            return

        if self.app is not None:
            self.app.log_message("Unrotate tiff file and perform image registration...")

        circlecenterfilename = os.path.join(self.DPFolder, "circlecenter.txt")
        if not os.path.exists(circlecenterfilename):
            self.log_message("Error: circlecenter.txt not found in DP folder.")
            return

        with open(circlecenterfilename, "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1]
            self.rotx = float(last_line.split()[0])
            self.roty = float(last_line.split()[1])

        self.log_message("Reading tiff file with tifffile...")
        frames = tifffile.imread(self.tifffilename)
        orig_shape = frames.shape
        self.log_message(f"Raw TIFF shape: {orig_shape}, dtype={frames.dtype}")

        if frames.ndim != 3:
            self.log_message(
                f"Error: Expected a 3D tiff stack [n_frames, height, width], got {frames.ndim}D."
            )
            return

        n_frames = frames.shape[0]
        angles, _, _ = get_frame_angles_from_rotary(
            self.tifffilename, self.relogfilename
        )

        if angles.size != n_frames:
            min_len = min(angles.size, n_frames)
            self.log_message(
                f"Warning: {angles.size} angles for {n_frames} frames; trimming to {min_len}."
            )
            frames = frames[:min_len]
            angles = angles[:min_len]
            n_frames = min_len

        self.unrotFrames = get_unrotate_crop_cv2(
            frames, angles, rotCenter=[self.rotx, self.roty]
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meanRegImg, self.regFrames = RegFrame(self.unrotFrames, device)

        self.display_regFrame()

        if self.app is not None:
            self.app.log_message("Perform Correlation Analysis...")

        meanstacks_path = os.path.join(self.DPFolder, "meanstacks.npy")
        if not os.path.exists(meanstacks_path):
            self.log_message("Error: meanstacks.npy not found in DP folder.")
            return

        meanstacks = np.load(meanstacks_path)
        ops = default_ops.default_ops()
        _, _, self.corrMatrix = compute_zpos_sp(meanstacks, self.regFrames, ops)
        self.corrMatrix = gaussian_filter1d(self.corrMatrix.copy(), 2, axis=0)

        maxrotangle = 30
        interval = maxrotangle // 3
        _, _, mean_zcorr = findFOV(
            meanstacks, self.meanRegImg, maxrotangle=maxrotangle
        )
        mean_zcorr_gs = gaussian_filter(mean_zcorr.copy(), 2)
        maxvalue = np.max(mean_zcorr_gs)
        maxindex = np.where(mean_zcorr_gs == maxvalue)

        fig = plt.figure()
        plt.imshow(mean_zcorr_gs, cmap="coolwarm", aspect="auto")
        plt.xticks(
            np.arange(0, 2 * maxrotangle + 1, interval),
            np.arange(-maxrotangle, maxrotangle + 1, interval),
        )
        plt.colorbar()
        plt.title(f"zcorr map, maxzplane={maxindex[0][0]}")
        plt.xlabel("rotation degree")
        plt.ylabel("stack index")
        plt.plot(maxindex[1][0], maxindex[0][0], "ro")
        fig.savefig(os.path.join(self.DPFolder, "maxcorrmeanframe.png"))
        plt.close(fig)

        self.display_corrMatrix()

    def display_regFrame(self):
        fig = plt.figure(figsize=(512 / 100, 512 / 100), dpi=100)
        plt.imshow(self.meanRegImg, cmap="gray")
        plt.axis("off")
        fig.savefig(os.path.join(self.DPFolder, "meanReg.png"))
        plt.close(fig)

        pixmap = QtGui.QPixmap(os.path.join(self.DPFolder, "meanReg.png"))
        self.reg_image.setPixmap(pixmap.scaled(
            self.reg_image.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        ))

    def display_corrMatrix(self):
        fig = plt.figure(figsize=(512 / 100, 256 / 100), dpi=100)
        nplanes, nframes = self.corrMatrix.shape

        gs = GridSpec(1, 2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.corrMatrix, aspect="auto", cmap="gray")
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Stack index")
        ax1.set_yticks(np.arange(0, nplanes, 5))
        ax1.set_yticklabels(np.arange(0, nplanes, 5) - int(nplanes / 2))
        ax1.axhline(y=nplanes / 2, color="r", linestyle="-")

        ax2 = fig.add_subplot(gs[0, 1])
        sumCorrMatrix = np.sum(self.corrMatrix, axis=1)
        ax2.plot(sumCorrMatrix, np.arange(0, nplanes), color="grey")
        ax2.set_xlabel("Sum of cc")
        ax2.set_yticks(np.arange(0, nplanes, 5))
        ax2.set_yticklabels(np.arange(0, nplanes, 5) - int(nplanes / 2))
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.axhline(y=nplanes / 2, color="r", linestyle="-")
        maxIndex = np.argmax(sumCorrMatrix)
        ax2.plot(sumCorrMatrix[maxIndex], maxIndex, "ro")

        shiftamount = maxIndex - int(nplanes / 2)
        ax2.text(
            0.5,
            0.1,
            f"Dft= {shiftamount}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            color="r",
        )

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        fig.savefig(os.path.join(self.DPFolder, "corrMatrix.png"))
        plt.close(fig)

        pixmap = QtGui.QPixmap(os.path.join(self.DPFolder, "corrMatrix.png"))
        self.corr_image.setPixmap(pixmap.scaled(
            self.corr_image.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        ))

        if self.app is not None:
            if shiftamount < 0:
                self.app.log_message(f"Move zforcus {-shiftamount} micrometers down")
            else:
                self.app.log_message(f"Move zforcus {shiftamount} micrometers up")
