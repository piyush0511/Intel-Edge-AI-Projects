
eval "$(conda shell.bash hook)"
conda activate vino

source /opt/intel/openvino/bin/setupvars.sh


if [[ $1 == "high" ]];
then
	python main.py --modelf /home/openvino_models/intel/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --modelp /home/openvino_models/intel/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 --modell /home/openvino_models/intel/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 --modelg /home/openvino_models/intel/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 #--flag yes
fi



if [[ $1 == "medium" ]];
then
	python main.py --modelf /home/openvino_models/intel/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --modelp /home/openvino_models/intel/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 --modell /home/openvino_models/intel/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 --modelg /home/openvino_models/intel/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 #--flag yes
fi

if [[ $1 == "low" ]];
then
	python main.py --modelf /home/openvino_models/intel/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --modelp /home/openvino_models/intel/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001 --modell /home/openvino_models/intel/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009 --modelg /home/openvino_models/intel/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002 #--flag yes
fi

