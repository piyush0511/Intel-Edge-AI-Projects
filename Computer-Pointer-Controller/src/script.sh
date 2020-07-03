
#conda activate vino
#source /opt/intel/openvino/bin/setupvars.sh

python main.py --modelf /home/openvino_models/intel/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --modelp /home/openvino_models/intel/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 --modell /home/openvino_models/intel/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 --modelg /home/openvino_models/intel/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 --flag yes


