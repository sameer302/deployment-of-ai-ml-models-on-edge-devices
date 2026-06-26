"""
yolo_ncnn_infer.py
==================
Lightweight YOLO11 inference for Raspberry Pi 5.
Uses the ncnn Python bindings directly — no ultralytics, no torch, no opencv-dnn.

Install on Pi:
  pip install ncnn opencv-python numpy pyyaml

Usage examples:
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source test.jpg
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source images_dir
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source video.mp4
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source usb0
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source picamera0 --resolution 640x480
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source picamera0 --resolution 1280x720 --display 640x480
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source usb0 --resolution 1280x720 --record

  # With metrics logging:
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source usb0 --log-latency --log-inference-latency --log-fps --log-inference-fps --csv-path metrics.csv
  python yolo_ncnn_infer.py --model yolo11n_ncnn_model --source usb0 --log-latency --csv-path /home/pi/logs/metrics.csv --frame-window-size 20
"""


# -------------------------Importing Libraries Code--------------------------------------

import os # done
import sys # done
import argparse # done
import glob
import time
import signal # done
import cv2
import numpy as np
import yaml # done
import ncnn # done

# ── METRICS: standard library imports for CSV writing and datetime formatting ──
import csv
import datetime
from collections import deque


 
#---------------------------------------------------------------------------------------- 






# ----------------------- Program Runtime Management Code (applicable for any inference script)-----------------------------------------
# Packages/libraries needed here:
# 1) signal


_shutdown_requested = False #  a global flag is created

def _request_shutdown(signum, _frame): # _frame is a conventional way to name a parameter that is internal or unused. 
'''
# This function allows the inference script to clean up files, cameras, CSVs, etc., 
before exiting instead of stopping abruptly.
'''
    global _shutdown_requested 
'''
# Here global indicates python that we should use global variable named _shutdown_requested and change it's value. 
# If we don't use global then inside a function, a variable is always treated like a local variable. 
# We need to explicitly define global only when we are modifying the value of that variable and not when we are just reading that value. 
'''
    print(f'\n[INFO] Signal {signum} received. Shutting down gracefully...')
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _request_shutdown) # kill <pid> --> SIGTERM
signal.signal(signal.SIGINT,  _request_shutdown) # Ctrl + C in terminal --> SIGINT

'''
# signal.signal(X, Y) is the function that tells python, "when signal X arrives, run function Y"
# Syntax: signal.signal(signal_number, handler_function), signal number and current execution frame are passed as arguments to the handler_function 
although frame parameter is mostly not used but API requires it. 
# signal is a message sent by OS to the process.

# Signal    Meaning                 signal_number
# SIGINT    Interrupt (Ctrl + C)        2
# SIGTERM   Terminate                   15    

signal.signal(signal.SIGINT, signal.SIG_DFL) # --> OS's default handling.
signal.signal(signal.SIGTERM, signal.SIG_IGN) # --> Ignoring a signal
'''

#-----------------------------------------------------------------------------------








# ------------------------- Argument Parsing Code (applicable for any inference script, the arguments may change)------------------
# Packages/libraries needed here:
# 1) argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model',      required=True,
                    help='Path to NCNN model folder  (contains model.ncnn.param, '
                         'model.ncnn.bin, metadata.yaml)')
parser.add_argument('--source',     required=True,
                    help='Image file, image folder, video file, usb<N>, or picamera<N>')
parser.add_argument('--thresh',     type=float, default=0.5,
                    help='Confidence threshold  (default: 0.5)')
parser.add_argument('--iou',        type=float, default=0.45,
                    help='NMS IoU threshold  (default: 0.45)')
parser.add_argument('--resolution',
                    help='Capture resolution WxH, e.g. 1280x720  (required for picamera/usb/record)')
parser.add_argument('--display',
                    help='Display window resolution WxH, e.g. 640x480  '
                         '(optional — defaults to capture resolution if not set)')
parser.add_argument('--record',     action='store_true',
                    help='Record output to demo1.avi  (requires --resolution)')

# ── METRICS: five new arguments for controlling what gets logged and where ──
parser.add_argument('--log-latency',          action='store_true',
                    help='Log end-to-end frame latency (capture → display) to CSV')
parser.add_argument('--log-inference-latency', action='store_true',
                    help='Log pure inference latency (ncnn forward pass only) to CSV')
parser.add_argument('--log-fps',              action='store_true',
                    help='Log end-to-end FPS (rolling window) to CSV. '
                         'Only valid for video/camera sources.')
parser.add_argument('--log-inference-fps',    action='store_true',
                    help='Log inference-only FPS (rolling window) to CSV. '
                         'Only valid for video/camera sources.')
parser.add_argument('--csv-path',             type=str, default=None,
                    help='Path to the output CSV file, e.g. metrics.csv or /home/pi/logs/metrics.csv. '
                         'Required if any --log-* flag is set.')
parser.add_argument('--frame-window-size',          type=int, default=30,
                    help='Rolling window size (number of frames) used to compute FPS. '
                         'Default: 30')

args = parser.parse_args()

#-----------------------------------------------------------------------------------










# ------------------------------Locating Model Files Code (typical for ncnn model format but can be changed as per other model format)---------------------------------------------
# Packages/libraries needed here:
# 1) os
# 2) sys


model_dir     = args.model
param_path    = os.path.join(model_dir, 'model.ncnn.param')
bin_path      = os.path.join(model_dir, 'model.ncnn.bin')
metadata_path = os.path.join(model_dir, 'metadata.yaml')

for fpath, name in [(param_path,    'model.ncnn.param'),
                    (bin_path,      'model.ncnn.bin'),
                    (metadata_path, 'metadata.yaml')]:
    if not os.path.exists(fpath):
        print(f'ERROR: {name} not found in "{model_dir}"')
        sys.exit(1)

#-----------------------------------------------------------------------------------








# ----------------------------Parse metadata.yaml Code (typical for ncnn format but can be changed as per other model format)-----------------------------------------------
# Packages/libraries needed here:
# 1) yaml


with open(metadata_path) as f:
    meta = yaml.safe_load(f) 
'''
    # reads the YAML file and converts it into a python dictionary named meta.
    # imgsz:
    # -640
    # -640
    # is YAML syntax for list.
'''

names_dict  = meta['names'] 
'''
extracts the value associated with key 'names' which is a dictionary in itself. 
So names_dict contains values like {0: 'person', 1: 'bicycle', …}
'''
labels      = [names_dict[i] for i in sorted(names_dict)] # sorts the keys and creates a list of class names
num_classes = len(labels)

imgsz   = meta.get('imgsz', [640, 640]) # if 'imgsz' key is missing then use [640, 640] by default
INPUT_H, INPUT_W = (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])

print(f'[INFO] {num_classes} classes | input size: {INPUT_W}x{INPUT_H}')
print(f'[INFO] Loading model from "{model_dir}" ...')

#-----------------------------------------------------------------------------------








# -----------------------------Load NCNN model (single Net instance, reused every frame)----------------------------------------------
# Packages/libraries used 
# 1) ncnn


net = ncnn.Net()
net.load_param(param_path)
net.load_model(bin_path)
print('[INFO] Model loaded.')

# ---------------------------------------------------------------------------







#-------------------------------Preprocessing Code----------------------------------------------------
# Packages/libraries used
# 1) cv2
# 2) ncnn

'''
What all values we need to infer from the training process/code of the model used by us, 
1) That is must for model to work:-
a) input height and input width

2) That might affect the performance of the model:-
a) what padding strategy was used during training (fill with constant color, replicate edge pixels or 
reflect the image around edges). 
b) mean_vals and norm_vals for each channel. 
c) image layout expected (RGB or BGR)
'''



def preprocess(frame: np.ndarray): 
'''
here the argument is frame and this syntax "frame: np.ndarray" is called type hint. 
It is not enforced at runtime by default so even if we write preprocess(5), it will be valid.
Its mainly for documentation and better code readability. 
We can use it for any data type like, "frame: int" or "frame: float", etc.

This is necessary becaue in future when we will use, "frame = cv2.imread(image.-jpg)"
here OpenCV image is stored as a NumPy array of type uint8 whose shape is (height, width, channels(BGR)).
so we can say that openCV internally uses NumPy. 
'''
    h, w = frame.shape[:2]
    scale  = min(INPUT_W / w, INPUT_H / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
'''
This ensures that our image fits inside the dimensions expected by the model while preserving aspect ratio.
'''

    pad_top    = (INPUT_H - new_h) // 2
    pad_bottom =  INPUT_H - new_h - pad_top # This is necessary for the case when (INPUT_H - new_h) is float. 
    pad_left   = (INPUT_W - new_w) // 2
    pad_right  =  INPUT_W - new_w - pad_left

    padded = cv2.copyMakeBorder(resized,
                                pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
'''
Here cv2.copyMakeBorder() needs to know how the new border pixels should be filled. 
The argument cv2.BORDER_CONSTANT means fill the added border with a constant value. The
constant value is specified by value=(114, 114, 114). Here we gave a tuple as argument 
instead of a single value because we need to fill the frame with three channel values. 

Other arguments are cv2.BORDER_REPLICATE or cv2.BORDER_REFLECT. 

What type of argument and for cv2.BORDER_CONSTANT what color to use for filling this depends
on training code. We should match training preprocessing as the model has learned to see such borders.

(114, 114, 114) gives a gray color. 

This section of preprocessing is called letterboxing.
'''

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)  
'''
OpenCV loads images as B G R but most deep learning models expect R G B hence the 
above command swaps channels. 
'''

    # from_pixels: HWC uint8 → ncnn.Mat (CHW float), pixel_type=PIXEL_RGB
    mat_in = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB,
                                  INPUT_W, INPUT_H)
'''
The purpose of this function is to convert the numpy ndarray image into an NCNN tensor (ncnn.Mat)

We give our numpy image (rgb) as argument which has shape (height, width, number of 
channels) and dtype (uint8). Also the layout of our image shape is HWC.

Then we tell that the image is in RGB format by giving the argument ncnn.Mat.PixelType.PIXEL_RGB.

Then we give input image height and width as arguments. 

We need to give all these details because from_pixels() is part of NCNN's C++ API design where
the function is meant to work with raw pixel buffers and so it can't access metadata like NumPy provides. 

After all these arguments the conversion that happens is, 
NumPy Array - Shape: (640, 640, 3) - Layout: HWC - Type: uint8
ncnn.Mat - Shape: (3, 640, 640) - Layout: CHW - Type: float

This is necessary because NCNN inference expects an object of type ncnn.Mat and not numpy.ndarray
'''
    # Normalise to [0, 1]: x = (x - mean) * norm  →  mean=0, norm=1/255
    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
    mat_in.substract_mean_normalize(mean_vals, norm_vals)

    return mat_in, scale, pad_left, pad_top
'''
We return these 4 values because they are used later to convert model outputs back to original coordinates.
For example, 
Original Image -> Resize by scale -> Add padding -> Model prediction -> Remove padding -> Divide by scale -> original coordinates
'''

#-----------------------------------------------------------------------------------------------












# -------------------------Postprocessing Code--------------------------------------------------
# Packages/libraries used
# 1) cv2


def postprocess(raw: np.ndarray, scale, pad_left, pad_top,
                orig_w, orig_h, conf_thresh, iou_thresh):
'''
This function takes inputs as,
raw -> direct ouput from NCNN Model
scale -> scaling factor used during preprocessing
pad_left, pad_top -> padding added during letterboxing
orig_w, orig_h -> original image dimensions
conf_thresh -> confidence threshold
iou_thresh -> IoU threshold for NMS
'''
    out = raw[0]                            # raw.shape = (1, 84, 8400) and out.shape = (84, 8400)
    box_data     = out[:4, :]               # box_data.shape = (4, 8400) contains row0 -> cx, row1 -> cy, row2 -> w, row3 -> h
    class_scores = out[4:4 + num_classes, :] # class_scores.shape = (80, 8400) contains person score, bicycle score, ..., toothbrush score

    # Vectorised best-class selection
    class_ids   = class_scores.argmax(axis=0)  # class_ids.shape = (8400,) finds the index/row number of class having max confidence score in each column
    confidences = class_scores.max(axis=0)     # confidences.shape = (8400,) finds the value of maximum confidence score in each column

'''
So after this instead of keeping all 80 class scores, we are just keeping highest class score
'''
    # Early filter — avoids looping over thousands of weak anchors
    mask = confidences >= conf_thresh
'''
if confidences = [0.91, 0.12, 0.83, 0.05, 0.30] and conf_thresh = 0.25 then, 
mask = [True, False, True, False, True]
'''
    if not mask.any(): # executes if there is no True value in mask
        return []

    box_data    = box_data[:, mask] # boolean indexing, this considers all rows and selects only those columns where mask=True
    class_ids   = class_ids[mask] # same as above
    confidences = confidences[mask] # same as above

    # cx,cy,w,h (padded-input space) → corner coords (original image space)
    cx, cy, bw, bh = box_data
    cx = (cx - pad_left) / scale
    cy = (cy - pad_top)  / scale
    bw /= scale
    bh /= scale
    
'''
cx → horizontal center coordinate of the box (distance in pixels from the left boundary of the image to the center of the box.)
cy → vertical center coordinate of the box (distance in pixels from the top boundary of the image to the center of the box.)
w  → width of the box
h  → height of the box
Above we are just reversing the operations that we did during pre-processing. 
'''

    xmins = np.clip(cx - bw / 2, 0, orig_w - 1).astype(int)
    ymins = np.clip(cy - bh / 2, 0, orig_h - 1).astype(int)
    xmaxs = np.clip(cx + bw / 2, 0, orig_w - 1).astype(int)
    ymaxs = np.clip(cy + bh / 2, 0, orig_h - 1).astype(int)

'''
# Above we are converting center formats to corners because YOLO predicts: (cx, cy, w, h) but OpenCV drawing functions need: (xmin, ymin, xmax, ymax)
# Also we are using np.clip to restrict the coordinates between 0 and original width - 1.
# Further .astype(int) converts the coordinates into integers. 
'''
    # cv2.dnn.NMSBoxes wants boxes as [x, y, width, height]
    boxes_xywh = [[int(x), int(y), int(x2 - x), int(y2 - y)]
                  for x, y, x2, y2 in zip(xmins, ymins, xmaxs, ymaxs)]
    conf_list  = confidences.tolist() # openCV prefers normal Python lists

    indices = cv2.dnn.NMSBoxes(boxes_xywh, conf_list, conf_thresh, iou_thresh) # performs NMS. Also we can remove the conf_thresh here with 0 as we already did it before and this step is redundant. 
    if len(indices) == 0:
        return []

    results = []
    for idx in indices.flatten():
        x, y, w, h = boxes_xywh[idx]
        results.append((x, y, x + w, y + h,
                        int(class_ids[idx]), conf_list[idx]))
    return results








# ---------------------------------------------------------------------------
# Bounding-box colours — Tableau-10 palette
# ---------------------------------------------------------------------------
BBOX_COLORS = [
    (164,120, 87), ( 68,148,228), ( 93, 97,209), (178,182,133), ( 88,159,106),
    ( 96,202,231), (159,124,168), (169,162,241), ( 98,118,150), (172,176,184),
]


















# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------
IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP'}
VID_EXTS = {'.avi','.mov','.mp4','.mkv','.wmv'}

img_source = args.source
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1]
    if ext in IMG_EXTS:
        source_type = 'image'
    elif ext in VID_EXTS:
        source_type = 'video'
    else:
        print(f'ERROR: Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'ERROR: Unrecognised source: {img_source}')
    sys.exit(1)

# ── METRICS: warn user if FPS logging is requested on image/folder sources ──
# FPS only makes sense for continuous streams; for static images it is meaningless.
if source_type in ('image', 'folder'):
    if args.log_fps or args.log_inference_fps:
        print('[WARNING] --log-fps and --log-inference-fps are only valid for '
              'video and live camera sources. These metrics will be skipped.')
        # Force-disable the flags so the rest of the code doesn't try to log them
        args.log_fps           = False
        args.log_inference_fps = False

# ── METRICS: validate that --csv-path is provided whenever any log flag is set ──
any_logging = args.log_latency or args.log_inference_latency or \
              args.log_fps or args.log_inference_fps
if any_logging and not args.csv_path:
    print('ERROR: --csv-path must be specified when any --log-* flag is set.')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Resolution / recording setup
# ---------------------------------------------------------------------------

# --- Capture resolution (what the camera grabs / frame is resized to) ---
resize = False
resW = resH = None
if args.resolution:
    resize = True
    resW, resH = (int(v) for v in args.resolution.split('x'))

# Picamera always needs an explicit capture resolution
if source_type == 'picamera' and not args.resolution:
    print('ERROR: --resolution is required for picamera source  (e.g. --resolution 640x480)')
    sys.exit(1)

# --- Display resolution (what is shown in the OpenCV window) ---
dispW = dispH = None
if args.display:
    dispW, dispH = (int(v) for v in args.display.split('x'))
elif args.resolution:
    # Default display = capture resolution
    dispW, dispH = resW, resH
# If neither is set (image/folder sources), display at native frame size

if args.record:
    if source_type not in ('video', 'usb'):
        print('ERROR: Recording only works for video/camera sources.')
        sys.exit(1)
    if not args.resolution:
        print('ERROR: Please specify --resolution to record.')
        sys.exit(1)
    # Record at capture resolution, not display resolution
    recorder = cv2.VideoWriter('demo1.avi',
                               cv2.VideoWriter_fourcc(*'MJPG'),
                               30, (resW, resH))

# ---------------------------------------------------------------------------
# Open source
# ---------------------------------------------------------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*'))
                 if os.path.splitext(f)[1] in IMG_EXTS]
    if not imgs_list:
        print(f'ERROR: No supported images found in "{img_source}"')
        sys.exit(1)
elif source_type in ('video', 'usb'):
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if resize:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# ---------------------------------------------------------------------------
# METRICS: CSV setup
# ---------------------------------------------------------------------------
# We only open/create the CSV file if at least one logging flag is active.
# The header row always includes 'timestamp', followed by whichever metric
# columns the user enabled.

csv_file    = None   # file handle, kept open for the duration of the loop
csv_writer  = None   # csv.writer instance

if any_logging:
    # Build the list of column headers based on which flags are active.
    # 'timestamp' is always first.
    csv_columns = ['timestamp']
    if args.log_latency:
        csv_columns.append('end_to_end_latency_ms')
    if args.log_inference_latency:
        csv_columns.append('inference_latency_ms')
    if args.log_fps:
        csv_columns.append('end_to_end_fps')
    if args.log_inference_fps:
        csv_columns.append('inference_fps')

    # Create parent directories if they don't exist, so the user can pass
    # a path like /home/pi/logs/metrics.csv without needing to mkdir first.
    csv_dir = os.path.dirname(args.csv_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    csv_file   = open(args.csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()
    print(f'[INFO] Metrics CSV will be written to: {args.csv_path}')
    print(f'[INFO] Logging will start after a 2-minute warm-up period.')

# ── METRICS: warm-up timer — record the absolute time when the loop starts.
# We will compare against this at every frame and skip logging for the
# first 120 seconds (2 minutes).
WARMUP_SECONDS = 120
loop_start_time = None   # will be set on the very first iteration

# ── METRICS: rolling windows for FPS calculation.
# Each deque stores (wall_clock_timestamp, latency_seconds) for the last
# `window_size` frames, separately tracked for e2e and inference.
# When the deque reaches window_size entries, we can compute:
#   FPS = window_size / (timestamp_of_last - timestamp_of_first)
WINDOW_SIZE = args.frame_window_size

# Deque of wall-clock timestamps (time.perf_counter) at the moment each
# frame *completed* the e2e pipeline (just before cv2.imshow).
e2e_window = deque(maxlen=WINDOW_SIZE)

# Deque of wall-clock timestamps at the moment each frame *completed*
# the ncnn forward pass.
infer_window = deque(maxlen=WINDOW_SIZE)

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
avg_frame_rate    = 0.0
frame_rate_buffer = []
FPS_AVG_LEN       = 200
img_count         = 0

while not _shutdown_requested:
    # ── METRICS: t_frame_start marks the very beginning of this frame's
    # pipeline — immediately after we enter the loop, before any capture.
    # This is the "start" timestamp for end-to-end latency.
    t_frame_start = time.perf_counter()

    # ── METRICS: record the absolute wall time for the first frame so we
    # can track when the 2-minute warm-up period ends.
    if loop_start_time is None:
        loop_start_time = time.time()

    # --- Load frame ---------------------------------------------------------
    if source_type in ('image', 'folder'):
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
        if frame is None:
            print(f'WARNING: Could not read {imgs_list[img_count-1]}, skipping.')
            continue

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of video. Exiting.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Camera read failed. Exiting.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Picamera read failed. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    orig_h, orig_w = frame.shape[:2]

    # --- Pre-process --------------------------------------------------------
    mat_in, scale, pad_left, pad_top = preprocess(frame)

    # --- Inference ----------------------------------------------------------
    # create_extractor() is lightweight — a thin stateless view over the net

    # ── METRICS: t_infer_start marks the beginning of the pure ncnn forward
    # pass. We record this right before ex.input so the clock starts the
    # moment we hand data to the network.
    t_infer_start = time.perf_counter()

    with net.create_extractor() as ex:
        ex.input("in0", mat_in)          # stage input tensor
        _, out0 = ex.extract("out0")     # run forward pass — this is where the computation happens

    # ── METRICS: t_infer_end marks the moment the ncnn forward pass is done.
    # inference_latency_s = t_infer_end - t_infer_start
    t_infer_end = time.perf_counter()
    inference_latency_s = t_infer_end - t_infer_start

    # out0 is an ncnn.Mat; convert to numpy then reshape to (1, 4+nc, anchors)
    raw = np.array(out0).reshape(1, 4 + num_classes, -1)

    # --- Post-process -------------------------------------------------------
    detections = postprocess(raw, scale, pad_left, pad_top,
                             orig_w, orig_h, args.thresh, args.iou)

    # --- Draw results -------------------------------------------------------
    object_count = 0
    for (xmin, ymin, xmax, ymax, class_id, conf) in detections:
        color = BBOX_COLORS[class_id % 10]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        label = f'{labels[class_id]}: {int(conf * 100)}%'
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, lh + 10)
        cv2.rectangle(frame,
                      (xmin, label_ymin - lh - 10),
                      (xmin + lw,  label_ymin + baseline - 10),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        object_count += 1

    if source_type in ('video', 'usb', 'picamera'):
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}',
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f'Objects: {object_count}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Resize to display resolution if it differs from capture resolution
    display_frame = frame
    if dispW and dispH and (dispW != frame.shape[1] or dispH != frame.shape[0]):
        display_frame = cv2.resize(frame, (dispW, dispH))

    # ── METRICS: t_frame_end marks the moment the fully-annotated frame is
    # ready to be shown — right before cv2.imshow. This is the "end"
    # timestamp for end-to-end latency.
    t_frame_end = time.perf_counter()
    e2e_latency_s = t_frame_end - t_frame_start

    cv2.imshow('YOLO11n NCNN', display_frame)

    # --- Window-close and key handling --------------------------------------
    wait_ms = 0 if source_type in ('image', 'folder') else 5
    key = cv2.waitKey(wait_ms) & 0xFF

    if key in (ord('q'), ord('Q')):
        print('[INFO] Q pressed. Exiting.')
        break
    elif key in (ord('s'), ord('S')):
        cv2.waitKey(0)
    elif key in (ord('p'), ord('P')):
        cv2.imwrite('capture.png', frame)
        print('Saved capture.png')

    # Check if the window was closed via the X button.
    # WND_PROP_VISIBLE returns 0 or negative when the window is gone.
    try:
        if cv2.getWindowProperty('YOLO11n NCNN', cv2.WND_PROP_VISIBLE) < 1:
            print('[INFO] Display window closed. Exiting.')
            break
    except cv2.error:
        print('[INFO] Display window closed. Exiting.')
        break

    if args.record:
        recorder.write(frame)

    # --- FPS bookkeeping (original) -----------------------------------------
    t_stop  = time.perf_counter()
    fps_now = 1.0 / max(t_stop - t_frame_start, 1e-9)
    if len(frame_rate_buffer) >= FPS_AVG_LEN:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps_now)
    avg_frame_rate = float(np.mean(frame_rate_buffer))

    # ── METRICS: rolling window update ──────────────────────────────────────
    # Push the wall-clock timestamp of this frame into each rolling window.
    # We use time.perf_counter() so the timestamps are consistent with the
    # latency values we already computed above.
    current_perf = time.perf_counter()
    e2e_window.append(current_perf)       # one entry per displayed frame
    infer_window.append(t_infer_end)      # one entry per inferenced frame

    # ── METRICS: compute rolling FPS once the window is full ────────────────
    # FPS = number_of_frames / (timestamp_newest - timestamp_oldest)
    # When the deque hits maxlen, the oldest entry is automatically dropped,
    # so this always reflects the last `WINDOW_SIZE` frames.
    e2e_fps   = None
    infer_fps = None

    if len(e2e_window) == WINDOW_SIZE:
        elapsed_e2e = e2e_window[-1] - e2e_window[0]
        if elapsed_e2e > 0:
            # We have WINDOW_SIZE frames spanning elapsed_e2e seconds,
            # so there are (WINDOW_SIZE - 1) intervals between them.
            e2e_fps = (WINDOW_SIZE - 1) / elapsed_e2e

    if len(infer_window) == WINDOW_SIZE:
        elapsed_infer = infer_window[-1] - infer_window[0]
        if elapsed_infer > 0:
            infer_fps = (WINDOW_SIZE - 1) / elapsed_infer

    # ── METRICS: CSV writing ─────────────────────────────────────────────────
    # Only log after the 2-minute warm-up period has elapsed.
    # We check this every frame, but writes only happen once warm-up is done.
    if any_logging and csv_writer is not None:
        elapsed_since_start = time.time() - loop_start_time

        if elapsed_since_start >= WARMUP_SECONDS:
            # Build the timestamp string in ISO-8601 format:
            # e.g.  2025-07-15 14:32:07.413
            # datetime.now() gives local time with microseconds; we format
            # it to millisecond precision for readability.
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # Build the row dict; only include columns that were requested.
            row = {'timestamp': ts}

            if args.log_latency:
                # Convert seconds → milliseconds and round to 3 decimal places
                row['end_to_end_latency_ms'] = round(e2e_latency_s * 1000, 3)

            if args.log_inference_latency:
                row['inference_latency_ms'] = round(inference_latency_s * 1000, 3)

            if args.log_fps:
                # e2e_fps is None until the rolling window is full; write
                # empty string in the meantime so the CSV stays well-formed.
                row['end_to_end_fps'] = round(e2e_fps, 3) if e2e_fps is not None else ''

            if args.log_inference_fps:
                row['inference_fps'] = round(infer_fps, 3) if infer_fps is not None else ''

            csv_writer.writerow(row)

            # Flush every frame so data is not lost if the process is killed.
            csv_file.flush()

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
net.clear()
if source_type in ('video', 'usb'):
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if args.record:
    recorder.release()

# ── METRICS: close the CSV file handle cleanly on exit ──
if csv_file is not None:
    csv_file.close()
    print(f'[INFO] Metrics saved to: {args.csv_path}')

cv2.destroyAllWindows()
