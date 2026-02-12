import os
import glob
import tifffile as tiff
import numpy as np
import re
import datetime as dt


def get_imaging_files(datafolder, namelist, readVRlogs=True):
    '''
    get the triple of data files
    Args:
        datafolder: the folder containing the data files
        namelist: a list of names, e.g., [00003, 00004, 00005]
    Returns:
        A dictionary, with each element as a list of two files (readVRlogs=False) or three files (readVRlogs=True)
        1, tiff file
        2, RELog file
        3, VRLog file
    '''
    #get all the tiff files 
    tifffiles = glob.glob(datafolder + "/*.tif")
    #remove tifffiles with key words "stack"
    tifffiles = [x for x in tifffiles if "stack" not in x]
    #then only keep tiff files with the names in namelist
    tifffiles = [x for x in tifffiles if any(y in x for y in namelist)]
    #sort the tifffiles by the number in the file name
    tifffiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    #get all the RElog files under the parent folder begin with 'RE' and with an extension of .txt
    RElogfiles = glob.glob(datafolder + "/RE*.txt")
    #remove RElogfiles with key words "stack"
    RElogfiles = [x for x in RElogfiles if "stack" not in x]
    
    #get all the VRlog files under the parent folder begin with numbers and with an extension of .txt
    VRlogfiles = glob.glob(datafolder + "/[0-9]*.txt")

    #pair the tiff files and RElog files together which share the same key word
    allfiles = []
    for tifffile in tifffiles:
        #extract the key word from the tifffile
        #for example, '/home/zilong/Desktop/2D2P/Data/183_25072023/25072023_00005.tif' then extract '00005'
        key = tifffile.split("/")[-1].split(".")[0].split("_")[-1]

        #find the RElogfile containing the key word
        RElogfile = [x for x in RElogfiles if key in x][0]
        if readVRlogs:
            #find the VRlogfile containing the key word
            VRlogfile = [x for x in VRlogfiles if key in x][0]
            #pair the tifffile and RElogfile together
            pair = [tifffile, RElogfile, VRlogfile]
        else:
            pair = [tifffile, RElogfile]
            
        #append the pair to allfiles
        allfiles.append(pair)
        
    return allfiles

def get_rotary_center(centerfile):
    '''
    get the rotary center from the centerfile
    Args:
        centerfile: the file containing the rotary center
    Returns:
        the rotary center 
    '''
    
    with open(centerfile, "r") as f:
        # read the last row
        last_line = f.readlines()[-1]
        # assign the x and y coordinates to self.rotx and self.roty
        rotx = float(last_line.split()[0])
        roty = float(last_line.split()[1])
    
    rotCenter = [rotx, roty]
    
    return rotCenter



_num_re = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")


def _parse_value(v: str):
    v = v.strip()

    if v == "[]":
        return []
    if v == "{}":
        return {}

    # Bracket array like: [2023  7 27 16  9 18.354]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if inner == "":
            return []
        parts = re.split(r"[\s,]+", inner)
        out = []
        for p in parts:
            if not p:
                continue
            if _num_re.match(p):
                out.append(int(p) if re.match(r"^[+-]?\d+$", p) else float(p))
            else:
                out.append(p)
        return out

    if _num_re.match(v):
        return int(v) if re.match(r"^[+-]?\d+$", v) else float(v)

    return v


def parse_scanimage_description(desc: str) -> dict:
    """
    Parse ScanImage per-frame ImageDescription text into a dict.
    Only parses the first block before '---'.
    """
    d = {}
    block = desc.split('---', 1)[0]
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith('%') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        d[k.strip()] = _parse_value(v)
    return d


def _epoch_list_to_datetime(epoch_list):
    """
    epoch_list format: [Y, M, D, h, m, s.sss]
    """
    if not isinstance(epoch_list, (list, tuple)) or len(epoch_list) < 6:
        raise ValueError(f"Invalid epoch format: {epoch_list}")

    Y, M, D, h, m, sec = epoch_list[:6]
    sec_float = float(sec)
    sec_int = int(sec_float)
    usec = int(round((sec_float - sec_int) * 1_000_000.0))

    # handle rounding overflow (e.g. 18.9999996 -> 19.000000)
    if usec >= 1_000_000:
        sec_int += 1
        usec -= 1_000_000

    return dt.datetime(int(Y), int(M), int(D), int(h), int(m), int(sec_int), usec)


def get_scanimage_frame_times(path_tif: str):
    """
    Read a ScanImage TIFF and return per-frame START and ACQUIRE times.

    Returns
    -------
    frame_start_rel_sec : (N,) float64
        Frame START times relative to acquisition start (frameTimestamps_sec)

    frame_start_wallclock : (N,) datetime64[ns]
        Wall-clock frame START times (epoch + frameTimestamps_sec)

    frame_acquire_rel_sec : (N,) float64
        Frame ACQUIRE times approximated as next frame's START time

    frame_acquire_wallclock : (N,) datetime64[ns]
        Wall-clock frame ACQUIRE times
    """

    ts = []
    epoch_list = None

    with tiff.TiffFile(path_tif) as tf:
        for i, page in enumerate(tf.pages):
            desc = page.description

            if isinstance(desc, dict):
                dd = desc.get("Description", desc)
            elif isinstance(desc, str):
                dd = parse_scanimage_description(desc)
            else:
                raise TypeError(f"Page {i}: unexpected description type: {type(desc)}")

            if epoch_list is None:
                if "epoch" not in dd:
                    raise KeyError(f"Page {i}: missing epoch")
                epoch_list = dd["epoch"]

            if "frameTimestamps_sec" not in dd:
                raise KeyError(f"Page {i}: missing frameTimestamps_sec")

            ts.append(float(dd["frameTimestamps_sec"]))

    frame_start_rel_sec = np.asarray(ts, dtype=np.float64)

    # ---- wall-clock START times ----
    Y, M, D, h, m, sec = epoch_list[:6]
    sec_i = int(sec)
    usec = int(round((sec - sec_i) * 1_000_000))
    epoch_dt = dt.datetime(int(Y), int(M), int(D), int(h), int(m), sec_i, usec)
    epoch64 = np.datetime64(epoch_dt, "ns")

    frame_start_wallclock = epoch64 + (frame_start_rel_sec * 1e9).astype("timedelta64[ns]")

    # ---- ACQUIRE times: shift by one frame ----
    frame_acquire_rel_sec = np.empty_like(frame_start_rel_sec)
    frame_acquire_rel_sec[:-1] = frame_start_rel_sec[1:]

    if frame_start_rel_sec.size >= 2:
        dt_last = frame_start_rel_sec[-1] - frame_start_rel_sec[-2]
    else:
        dt_last = 0.0

    frame_acquire_rel_sec[-1] = frame_start_rel_sec[-1] + dt_last

    frame_acquire_wallclock = epoch64 + (frame_acquire_rel_sec * 1e9).astype("timedelta64[ns]")

    # return (
    #     frame_start_rel_sec,
    #     frame_start_wallclock,
    #     frame_acquire_rel_sec,
    #     frame_acquire_wallclock,
    # )

    return (
        frame_acquire_rel_sec,
        frame_acquire_wallclock,
    )

def read_rotary_log(path_txt, tA_wall=None, window_sec=1.0):
    """
    Read rotary log with columns:
    DateTime  MonotonicSec  AngleDeg
    Returns: (dt_list, mono_list, angle_list)
    """
    dt_list = []
    mono_list = []
    angle_list = []

    # optional time window filter
    t_min = None
    t_max = None
    if tA_wall is not None:
        tA_wall_arr = np.asarray(tA_wall)
        if tA_wall_arr.size > 0:
            if np.issubdtype(tA_wall_arr.dtype, np.datetime64):
                # convert numpy datetime64 -> python datetime
                def _dt64_to_dt(x):
                    ns = x.astype("datetime64[ns]").astype("int64")
                    return dt.datetime(1970, 1, 1) + dt.timedelta(microseconds=ns / 1000.0)

                t_min = _dt64_to_dt(tA_wall_arr.min())
                t_max = _dt64_to_dt(tA_wall_arr.max())
            else:
                # assume list of datetime objects
                t_min = min(tA_wall)
                t_max = max(tA_wall)

            if window_sec is None:
                window_sec = 0.0

            t_min = t_min - dt.timedelta(seconds=float(window_sec))
            t_max = t_max + dt.timedelta(seconds=float(window_sec))

    with open(path_txt, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            dt_str = parts[0] + " " + parts[1]
            mono_str = parts[2]
            angle_str = parts[3] if len(parts) > 3 else ""

            try:
                t = dt.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    t = dt.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue

            if t_min is not None and (t < t_min or t > t_max):
                continue

            try:
                mono = float(mono_str)
            except ValueError:
                continue

            if angle_str == "" or angle_str.lower() == "nan":
                angle = np.nan
            else:
                try:
                    angle = float(angle_str)
                except ValueError:
                    angle = np.nan

            dt_list.append(t)
            mono_list.append(mono)
            angle_list.append(angle)

    return dt_list, mono_list, angle_list


def get_frame_angles_from_rotary(tif_path, rotary_log_path, window_sec=1.0):
    """
    For each frame in tif_path, return nearest rotary angle from rotary_log_path.

    Returns
    -------
    angle_at_tA : (N,) float64
        Angle per frame, nearest neighbor matched by wall-clock time.
    tA_rel : (N,) float64
        Frame acquire times relative to acquisition start (seconds).
    tA_wall : (N,) datetime64[ns]
        Frame acquire wall-clock timestamps.
    """
    tA_rel, tA_wall = get_scanimage_frame_times(tif_path)
    dt_list, _, angle_list = read_rotary_log(rotary_log_path, tA_wall=tA_wall, window_sec=window_sec)

    angle_arr = np.asarray(angle_list, dtype=float)
    dt_arr = np.array(dt_list, dtype="datetime64[ns]")

    if len(tA_wall) == 0:
        angle_at_tA = np.array([], dtype=float)
    else:
        idx = np.searchsorted(dt_arr, tA_wall, side="left")
        idx = np.clip(idx, 0, len(dt_arr) - 1)

        prev_idx = np.clip(idx - 1, 0, len(dt_arr) - 1)
        use_prev = (idx > 0) & (
            np.abs(tA_wall - dt_arr[prev_idx]) <= np.abs(tA_wall - dt_arr[idx])
        )
        idx[use_prev] = prev_idx[use_prev]
        angle_at_tA = angle_arr[idx]

    return angle_at_tA, tA_rel, tA_wall

    
    
    
