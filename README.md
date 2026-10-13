# Deployment-of-AI-ML-Models-on-Edge-Devices

## quick start guide

- Clone the repo in your local folder.
- Open the scripts folder, navigate inside to the deployment setting you want to execute.
- Open the bash script and modify the arguments as per need or let them be as is. Refer to readme.md fo this particular deployment setting from the deployment_pipelines folder. 
- Finally run the bash script using the command `bash run_inference.sh`.

## aiml_models

- This folder acts like a repository for AIML models that we will use to run inference on edge devices.
- It is partitioned into two sub folders namely pre-built models, this contains models which are readily available on online repositories and other is custom models, which contains models which are modified/trained for a particular application by me or my lab. 
- Then we have sub-folders on the basis of AIML application domain such as computer-vision, natural language processing, speech processing , etc. 
- Then inside each domain we have folders for particular task being performed such as in computer vision domain we have object classification, detection, segmentation, etc. 
- Further for each task we have sub-folders to classify the model files on the basis of processor hardware on which they will be executed. For example for object detection I have two sub-folders named cpu and hailo8_npu.
- Finally for each processor hardware, inside we have the necessary model files which need to be used whhile runing the inference. For the same processor hardware we may have more than one model format. For e.g., we have various model formats to un inference on CPU such as .pt, onnx, ncnn, etc. 

## performance_analysis

- Removed from this repo — performance profiling (platform-level metrics like CPU, memory, thermal, power, and accelerator monitoring) now lives in the separate `performance-analysis` repo. If a deployment pipeline here needs performance profiling, use the profiling script from that repo rather than adding one here.

## deployment_pipelines

- In this folder first we navigate to the folder representing the processor on which we run the inference for e.g., cpu, npu, etc. Then we navigate to the folder representing the host device on which we are running the inference for e.g., raspberry pi 5.
- Then inside each deployment setting we will have a setup.sh file to ensure relevant hardware settings to run the inference, we will have requirements.txt file to note down the necessary dependencies, we will have the inference code file named as (model_name)_(model_format)_inference.py and finally we wil have the readme.md file which will contain information about this particular deployment pipeline. 

## scripts

- In this folder we navigate down to each deployment setting such as cpu and then inside it for raspberry pi 5
- For each of these setting we then have a bash script which contains code to run inference for different scenarios as mentioned in the corresponding bash script. 
 
## Future Tasks
- Add relative paths everywhere so that after cloning people don't face problem. 
- Add more system baseline settings
- Add more performance metrics to log
- Add setup_env.sh to every deployment pipeline in which I shpuld mention about what exact hardware component to use with its specifications and operating mode. 
- automatic graph generation
- thermal analysis
- power analysis
- CPU-core pinning
- benchmark reproducibility scoring
