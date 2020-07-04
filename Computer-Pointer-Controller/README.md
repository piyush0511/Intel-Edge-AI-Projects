# Computer Pointer Controller

Control mouse cursor by gaze estimation using OpenVino models 

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.**6** |
| OpenVino Version: |  2020.**2** |
| Models Required: |face-detection-adas-binary-0001   <br /><br />landmarks-regression-retail-0009 <br /><br /> head-pose-estimation-adas-0001 <br /><br />gaze-estimation-adas-0002|
| Hardware Used: |  Intel CPU i5 2nd gen|
| Environment Used: |  VMware: Ubuntu 18.1|

## Project Set Up and Installation
```
├── bin
│   └──  demo.mp4
├── gaze.log
├── models
├── README.md
├── requirements.txt
└── src
    ├── faced.py
    ├── landed.py
    ├── gazed.py
    ├── posed.py
    ├── main.py
    ├── input_feeder.py
    ├── model.py
    └── mouse_controller.py
```
## Demo
After you clone the repo, you need to install the dependecies using this command
```
pip3 install -r requirements.txt
```
After that you need to download OpenVino required models using `model downloader`.
* face-detection-adas-binary-0001
* landmarks-regression-retail-0009 
* head-pose-estimation-adas-0001 
* gaze-estimation-adas-0002

## Documentation
```
usage: main.py [-h] --modelf MODELF --modelp MODELP --modell MODELL --modelg
               MODELG [--flag FLAG] [--device DEVICE] [--video VIDEO]
               [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --modelf MODELF       Face Model
  --modelp MODELP       Head Pose Model
  --modell MODELL       Landmarks Model
  --modelg MODELG       Gaze Model
  --flag FLAG           Display Face (if yes type 'yes')
  --device DEVICE       Device
  --video VIDEO         Path to the video file
  --threshold THRESHOLD
                        Threshold

```
## Run command: 
```
python main.py --modelf /home/openvino_models/intel/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --modelp /home/openvino_models/intel/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 --modell /home/openvino_models/intel/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 --modelg /home/openvino_models/intel/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 #--flag yes
```

## Benchmarks
### Results for DEVICE = CPU
| Factor/Model       | All Models |
|--------------------|---------------|
|Load Time FP32      |  2565ms        |
|Load Time FP16      |  2366ms           | 
|Load Time FP16-INT8 |  1454ms           |
||||||
|Inference Time FP32 | 85.5ms         |
|Inference Time FP16 | 82ms            |
|Inference Time FP16-INT8| 79ms        |
||||||

## Results
* Load time for models with FP32 is less than FP16 and the same for FP16 models is less than INT8. 
* Inference time for models with FP32 is larger than FP16 and  Inference time for FP16 models is larger than INT8. 
