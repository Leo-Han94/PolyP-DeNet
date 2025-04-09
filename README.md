# PolyP-DeNet:An Efficient Object Detection Network for Early Polyp Detection
## Introduction
Early detection of gastrointestinal polyps is critical for preventing and treating colorectal cancer. However, object detection methods still face challenges in accurately detecting small polyps, delineating lesion boundaries, and extracting discriminative features from complex backgrounds. This paper presents an improved polyp detection network model (Polyp-DeNet) based on the You Only Look Once (YOLO) model to address these limitations. A novel feature pyramid network (FPN) is designed for this model to enable optimal adaptive multiscale feature fusion. Moreover, the multiscale feature extraction ability of the model is increased by improving the feature extraction module for the backbone layer. The enhanced detection head of the baseline model, combined with a specific loss function, effectively increases the localization precision of the model and reduces false alarms in challenging scenarios such as localized occlusion and varying lighting conditions. Extensive experiments show that Polyp-DeNet achieves superior performance to that of common YOLO variants and existing polyp detection methods, demonstrating significant improvements in key metrics. In addition, Polyp-DeNet demonstrates excellent generalizability on several publicly available polyp datasets, highlighting its potential as a reliable technological solution for intelligent screening in gastrointestinal endoscopy and advancing the use of artificial intelligence in medical image analysis.
## Document
### Experimental Environment
✅ Python 3.10.14  
✅ PyTorch 2.3.1  
✅ NVIDIA RTX 4090
```bash
pip install pypi
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
```
### Train
You can choose PolyP-DeNet's model in PolyP-DeNet.yaml
```bash
from ultralytics import YOLO

# Load a model
model = YOLO("WildB-YoLo.yaml") # pase model

# Train the model
model.train(
                data='data.yaml', # pase your dataset
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4, 
                optimizer='SGD', 
                project='runs/train',
                name='exp',
)
```
## PolyP-DeNet Dataset
```bash
@misc{
      polyp-2_dataset,
      title = { polyp 2 Dataset },
      type = { Open Source Dataset },
      author = { Salam },
      howpublished = { \url{ https://universe.roboflow.com/salam-rndvd/polyp-2 } },
      url = { https://universe.roboflow.com/salam-rndvd/polyp-2 },
      journal = { Roboflow Universe },
      publisher = { Roboflow },
      year = { 2023 },
      month = { dec },
      note = { visited on 2025-04-09 },
}
```
