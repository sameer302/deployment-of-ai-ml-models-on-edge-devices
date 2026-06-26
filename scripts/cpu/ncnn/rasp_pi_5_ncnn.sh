
CURRENT_DIR=$(pwd)
cd /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/deployment_pipelines/cpu/raspberry_pi_5/ncnn
source venv/bin/activate

# Set the duration for running the inference script (format: Xm for minutes, Xs for seconds)
# Examples: 7m (7 minutes), 30s (30 seconds), 2h (2 hours)
TIME_DURATION="7m"

# YOLOV11 inference using ncnn format

# The script will run for the specified TIME_DURATION and then automatically stop.

# Normally timeout acts as an intermediate process between terminal and the python script. So if we want to send any signal from terminal to the python process we should use
# --foreground argument, this makes the python process to run in foreground and accept any signal coming from OS. If not used then timeout process will recieve signal from OS
# and it will consume it while the signal won't reach the python process. 

# After the specified TIME_DURATION, timeout will send the SIGTERM signal to the process which is by default but we can also change it.

timeout --foreground --signal=SIGTERM $TIME_DURATION python yolo11_ncnn_inference.py \
    --model /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/aiml_models/pre-built_models/vision/object_detection/cpu/yolo11n_ncnn_model \
    --source picamera0 \
    --thresh 0.5 \
    --iou 0.45 \
    --resolution 1296x972 \
    --log-latency \
    --log-inference-latency \
    --log-fps \
    --log-inference-fps \
    --csv-path /home/sameer/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/deployment_pipelines/cpu/raspberry_pi_5/ncnn/yolo11_ncnn_metrics_temp.csv \
    # --frame-window-size 30
    # --display 640x360 \
    # --record

deactivate
cd $CURRENT_DIR
