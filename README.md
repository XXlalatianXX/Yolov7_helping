# Yolov7_helping
I use Yolov7 from WongKinYiu in github. this is for another helping you can use it easy

================== Begin ====================
Yolov7 : https://github.com/WongKinYiu/yolov7
Youtube : https://www.youtube.com/watch?v=4na_P6_7hMo
Youtube : https://www.youtube.com/watch?v=n2mupnfIuFY ==> new

1. Download yolov7 : git clone https://github.com/WongKinYiu/yolov7.git  => to folder
2. anaconda prompt : conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
	=> install pytorch for cuda 11.3
3. Come into C:\yolo\yolov7
	: conda create -n yoloV7 python=3.10			[Python 3.10.9 เราใช้]
4. same prompt
	: conda activate yoloV7
5. install requirements
	: pip install -r "requirements.txt"

	* in anaconda
	: import torch
	: torch.cuda.is_available()	=> If -> True -> can use ^-^
	If False
		: pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
		: pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
	|| It will can use

6.Inference on Image34
	: python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source ./input/person2.jpg
	: python detect.py --weights yolov7.pt --conf 0.1 --img-size 640 --source images.jpg

7.Inference on Video
	: python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source input/test.mp4

	: python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name 
	
	** if it's not show boundbox. => change the value to; half = False in line 31 of detect.py, it works fine

================================================================================
================================================================================
Youtube : https://www.youtube.com/watch?v=-QWxJ0j9EY8&list=LL&index=4&t=29s

install cuda do everything like above first clip ^^ Not this clip

1. Train
	: python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7-custom.yaml --name yolov7-custom --weights yolov7.pt



||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
==================   new   =========================
1. Download yolov7 : git clone https://github.com/WongKinYiu/yolov7.git  => to folder
2. anaconda prompt : conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
	=> install pytorch for cuda 11.3
3. Come into C:\yolo\yolov7
	: conda create -n yoloV7 python=3.10			[Python 3.10.9 เราใช้]
4. same prompt
	: conda activate yoloV7
5. install requirements
	: pip install -r "requirements.txt"

	: # CUDA 11.3
	  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

	** Before this process => You have to install cuda & cudnn & opencv completely

6.In main folder must already have => yolov7.pt & picture
	: python detect.py --weights yolov7.pt --conf 0.4 --img-size 640 --source images.jpg

