import os
import os
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major")

import torch
from suite2p.registration import zalign


from utils_image import get_unrotate_crop_cv2, RegFrame
from utils_io import load_buffer_frames_and_angles


class _ZDriftWorker(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object, object, object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, tifffilename, timefilename, relogfilename, datafolder):
        super().__init__()
        self._tifffilename = tifffilename
        self._timefilename = timefilename
        self._relogfilename = relogfilename
        self._datafolder = datafolder

    @QtCore.pyqtSlot()
    def run(self):
        try:
            circlecenterfilename = os.path.join(self._datafolder, "circlecenter.txt")
            if not os.path.exists(circlecenterfilename):
                raise FileNotFoundError("circlecenter.txt not found in DataProcessFolder.")

            with open(circlecenterfilename, "r", encoding="utf-8") as f:
                last_line = f.readlines()[-1]
                rotx = float(last_line.split()[0])
                roty = float(last_line.split()[1])

            self.log.emit("Reading tiff file and rotary angles...")
            frames, angles = load_buffer_frames_and_angles(
                self._tifffilename, self._timefilename, self._relogfilename
            )
            orig_shape = frames.shape
            self.log.emit(f"Raw TIFF shape: {orig_shape}, dtype={frames.dtype}")

            if frames.ndim != 3:
                raise ValueError(
                    f"Expected a 3D tiff stack [n_frames, height, width], got {frames.ndim}D."
                )

            n_frames = frames.shape[0]
            if angles.size != n_frames:
                raise ValueError(f"{angles.size} angles for {n_frames} frames; aborting.")

            unrot_frames = get_unrotate_crop_cv2(
                frames, angles, rotCenter=[rotx, roty]
            )
            mean_reg_img = unrot_frames.mean(axis=0)

            meanstacks_path = os.path.join(self._datafolder, "meanstacks.npy")
            if not os.path.exists(meanstacks_path):
                raise FileNotFoundError("meanstacks.npy not found in DataProcessFolder.")

            meanstacks = np.load(meanstacks_path)
            if isinstance(meanstacks, np.ndarray) and meanstacks.ndim == 3:
                meanstacks = [meanstacks[z] for z in range(meanstacks.shape[0])]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            corr_matrix = zalign.register_to_zstack(
                f_align_in=unrot_frames,
                refImgs=meanstacks,
                nonrigid=True,
                device=device,
            )

            self.finished.emit(unrot_frames, mean_reg_img, corr_matrix)
        except Exception as exc:
            self.error.emit(str(exc))


class QtZDriftProcessor(QtWidgets.QWidget):
    def __init__(self, folder=None, app=None, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.app = app

        self.tifffilename = None
        self.relogfilename = None
        self.timefilename = None
        self.DataProcessFolder = None

        self.meanRegImg = None
        self.closestStackImg = None
        self.regFrames = None
        self.corrMatrix = None
        self.unrotFrames = None
        self._worker_thread = None
        self._worker = None
        self._auto_timer = None
        self._last_auto_key = None
        self._pending_auto_key = None
        self._pending_auto_since = None
        self._auto_delay_sec = 3.0
        self._manual_override = False

        self._build_ui()
        self._start_auto_timer()

    def _build_ui(self):
        layout = QtWidgets.QGridLayout(self)

        self.import_tiff_btn = QtWidgets.QPushButton("Import buffered tiff")
        self.import_tiff_btn.clicked.connect(self.import_tiff_buffer)
        layout.addWidget(self.import_tiff_btn, 0, 0)

        self.import_relog_btn = QtWidgets.QPushButton("Import buffered log")
        self.import_relog_btn.clicked.connect(self.import_RElog_buffer)
        layout.addWidget(self.import_relog_btn, 1, 0)

        self.corr_btn = QtWidgets.QPushButton("Correlation Analysis")
        self.corr_btn.clicked.connect(self.correlationanalysis)
        layout.addWidget(self.corr_btn, 0, 1, 2, 1)

        self.auto_detect_chk = QtWidgets.QCheckBox("Auto Detect & Run")
        self.auto_detect_chk.setChecked(True)
        self.auto_detect_chk.stateChanged.connect(self._on_auto_toggle)
        layout.addWidget(self.auto_detect_chk, 2, 0, 1, 2)

        self.reg_image = pg.ImageView(view=pg.PlotItem())
        self.reg_image.ui.roiBtn.hide()
        self.reg_image.ui.menuBtn.hide()
        self.reg_image.getView().setTitle("Mean Reg Image")
        layout.addWidget(self.reg_image, 4, 0, 1, 1)

        self.stack_image = pg.ImageView(view=pg.PlotItem())
        self.stack_image.ui.roiBtn.hide()
        self.stack_image.ui.menuBtn.hide()
        self.stack_image.getView().setTitle("Closest Z-Stack")
        layout.addWidget(self.stack_image, 4, 1, 1, 1)

        self.corr_image = pg.ImageView(view=pg.PlotItem())
        self.corr_image.ui.roiBtn.hide()
        self.corr_image.ui.menuBtn.hide()
        layout.addWidget(self.corr_image, 5, 0, 1, 2)

    def log_message(self, message):
        if self.app is not None:
            self.app.log_message(message)

    def set_folder(self, folder):
        self.folder = folder
        self._manual_override = False

    def import_tiff_buffer(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a tiff file", self.folder or "", "tiff files (*.tif);;all files (*.*)"
        )
        if not filename:
            return
        self._manual_override = True
        self.tifffilename = filename
        self.log_message(f"Imported tiff file: {self.tifffilename}")

        self.DataProcessFolder = os.path.join(
            os.path.dirname(self.tifffilename), "DataProcessFolder"
        )

    def import_RElog_buffer(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select a time.txt file", self.folder or "", "txt files (*.txt);;all files (*.*)"
        )
        if not filename:
            return
        self._manual_override = True
        self.timefilename = filename
        self.log_message(f"Imported time.txt file: {self.timefilename}")

    def _start_auto_timer(self):
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(2000)
        self._auto_timer.timeout.connect(self._auto_detect_files)
        self._auto_timer.start()

    def _on_auto_toggle(self):
        if self.auto_detect_chk.isChecked():
            self._manual_override = False
            self._pending_auto_key = None
            self._pending_auto_since = None

    def _get_num_stacks(self):
        if not self.DataProcessFolder:
            return None
        meanstacks_path = os.path.join(self.DataProcessFolder, "meanstacks.npy")
        if not os.path.exists(meanstacks_path):
            return None
        try:
            meanstacks = np.load(meanstacks_path)
        except Exception:
            return None
        if not isinstance(meanstacks, np.ndarray):
            return None
        if meanstacks.ndim == 2:
            return 1
        if meanstacks.ndim == 3:
            return int(meanstacks.shape[0])
        return None

    def _corr_as_plane_frame(self):
        if self.corrMatrix is None:
            return None
        corr = np.asarray(self.corrMatrix)
        if corr.ndim != 2:
            return corr

        nstacks = self._get_num_stacks()
        if nstacks is None:
            return corr

        # Normalize to [n_stacks, n_frames] for plotting and shift readout.
        if corr.shape[0] == nstacks and corr.shape[1] != nstacks:
            return corr
        if corr.shape[1] == nstacks and corr.shape[0] != nstacks:
            return corr.T
        return corr

    def _pick_latest(self, paths):
        if not paths:
            return None
        return max(paths, key=lambda p: p.stat().st_mtime)

    def _auto_detect_files(self):
        if not self.auto_detect_chk.isChecked():
            return
        if self._manual_override:
            return
        if self._worker_thread is not None and self._worker_thread.isRunning():
            return
        if not self.folder or not os.path.exists(self.folder):
            return

        folder = Path(self.folder)
        tif_candidates = list(folder.glob("online_grab_*.tif")) + list(
            folder.glob("online_grab_*.tiff")
        )
        txt_candidates = list(folder.glob("online_grab_*_time.txt"))

        tiff_path = self._pick_latest(tif_candidates)
        time_path = self._pick_latest(txt_candidates)
        relog_path = Path(self.relogfilename) if self.relogfilename else None
        if relog_path is not None and not relog_path.exists():
            relog_path = None

        if tiff_path is None or time_path is None or relog_path is None:
            return

        auto_key = (str(tiff_path), str(time_path), str(relog_path))
        if auto_key == self._last_auto_key:
            return

        now = time.monotonic()
        if auto_key != self._pending_auto_key:
            self._pending_auto_key = auto_key
            self._pending_auto_since = now
            self.log_message(
                f"Detected new buffer files, waiting {self._auto_delay_sec:.0f}s before analysis..."
            )
            return

        if self._pending_auto_since is None or (now - self._pending_auto_since) < self._auto_delay_sec:
            return

        self._pending_auto_key = None
        self._pending_auto_since = None
        self._last_auto_key = auto_key
        self.tifffilename = str(tiff_path)
        self.timefilename = str(time_path)
        self.relogfilename = str(relog_path)
        self.DataProcessFolder = os.path.join(
            os.path.dirname(self.tifffilename), "DataProcessFolder"
        )
        self.log_message(
            f"Auto-detected files: {self.tifffilename}, {self.timefilename}, {self.relogfilename}"
        )
        self.correlationanalysis()

    def correlationanalysis(self):
        if not self.tifffilename or not self.timefilename or not self.relogfilename:
            self.log_message("Error: Please import tiff, time.txt, and RElog files.")
            return
        if not self.DataProcessFolder:
            self.log_message("Error: DataProcessFolder not set.")
            return
        if not os.path.exists(self.DataProcessFolder):
            self.log_message(
                "Error: DataProcessFolder not found. Run center detection first."
            )
            return

        if self.app is not None:
            self.app.log_message("Unrotate tiff file and perform image registration...")

        self.corr_btn.setEnabled(False)
        self._worker_thread = QtCore.QThread(self)
        self._worker = _ZDriftWorker(
            self.tifffilename, self.timefilename, self.relogfilename, self.DataProcessFolder
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log_message)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._on_thread_finished)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_worker_finished(self, unrot_frames, mean_reg_img, corr_matrix):
        self.unrotFrames = unrot_frames
        self.meanRegImg = mean_reg_img
        self.corrMatrix = corr_matrix
        self.display_meanFrame()
        self.display_closest_stack_frame()
        self.display_corrMatrix()
        self.corr_btn.setEnabled(True)

    def _on_worker_error(self, message):
        self.log_message(f"Error: {message}")
        self.corr_btn.setEnabled(True)

    def _on_thread_finished(self):
        self._worker_thread = None
        self._worker = None

    def display_meanFrame(self):
        if self.meanRegImg is None:
            return
        img = np.asarray(self.meanRegImg, dtype=np.float32)
        if img.ndim == 3:
            img = img.mean(axis=0)
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        p1, p99 = np.percentile(img, [1, 99])
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            p1 = float(np.min(img))
            p99 = float(np.max(img))
            if p99 <= p1:
                p99 = p1 + 1.0

        self.reg_image.setImage(img, autoLevels=False, levels=(p1, p99))
        self.reg_image.getView().autoRange()

    def display_closest_stack_frame(self):
        corr = self._corr_as_plane_frame()
        if corr is None or not self.DataProcessFolder:
            return

        meanstacks_path = os.path.join(self.DataProcessFolder, "meanstacks.npy")
        if not os.path.exists(meanstacks_path):
            return

        meanstacks = np.load(meanstacks_path)
        if not isinstance(meanstacks, np.ndarray):
            return
        if meanstacks.ndim == 2:
            meanstacks = meanstacks[np.newaxis, ...]
        if meanstacks.ndim != 3 or meanstacks.shape[0] == 0:
            return

        sumCorrByPlane = np.sum(corr, axis=1)
        maxIndex = int(np.argmax(sumCorrByPlane))
        maxIndex = max(0, min(maxIndex, meanstacks.shape[0] - 1))

        img = np.asarray(meanstacks[maxIndex], dtype=np.float32)
        if not np.isfinite(img).all():
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        p1, p99 = np.percentile(img, [1, 99])
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            p1 = float(np.min(img))
            p99 = float(np.max(img))
            if p99 <= p1:
                p99 = p1 + 1.0

        self.closestStackImg = img
        self.stack_image.setImage(img, autoLevels=False, levels=(p1, p99))
        self.stack_image.getView().autoRange()

    def display_corrMatrix(self):
        corr = self._corr_as_plane_frame()
        if corr is None or corr.ndim != 2:
            return

        fig = plt.figure(figsize=(10.0, 2.8), dpi=120)
        nplanes, nframes = corr.shape

        gs = GridSpec(1, 2, width_ratios=[5, 1.2])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(corr, aspect="auto", cmap="gray")
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Stack index")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        y_ticks = np.arange(0, nplanes, 5)
        center_idx = int((nplanes - 1) / 2)
        y_tick_labels = center_idx - y_ticks
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_tick_labels)
        ax1.axhline(y=(nplanes - 1) / 2.0, color="r", linestyle="-")
        ax1.set_xlim(-0.5, nframes - 0.5)

        ax2 = fig.add_subplot(gs[0, 1])
        sumCorrByPlane = np.sum(corr, axis=1)
        z_indices = np.arange(0, nplanes)
        ax2.plot(sumCorrByPlane, z_indices, color="grey")
        ax2.set_xlabel("Sum of correlation value")
        ax2.set_ylabel("Z-stack number")
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_tick_labels)
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax2.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax2.set_ylim(ax2.get_ylim()[::-1])
        maxIndex = np.argmax(sumCorrByPlane)
        ax2.plot(sumCorrByPlane[maxIndex], maxIndex, "ro")

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
        fig.savefig(os.path.join(self.DataProcessFolder, "corrMatrix.png"))
        plt.close(fig)

        corr_img = plt.imread(os.path.join(self.DataProcessFolder, "corrMatrix.png"))
        if corr_img.ndim == 3:
            corr_img = corr_img[:, :, 0]
        self.corr_image.setImage(corr_img, autoLevels=True)
        self.corr_image.getView().autoRange()

        if self.app is not None:
            if shiftamount < 0:
                self.app.log_message(f"Move zforcus {-shiftamount} micrometers down")
            else:
                self.app.log_message(f"Move zforcus {shiftamount} micrometers up")
