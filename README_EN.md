<div align="center">
<img src="docs/icon1.png" width="250"/>

[![PyPI Version](https://img.shields.io/pypi/v/cvglue)](https://pypi.org/project/cvglue/) 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**English** | [**中文简体**](./README.md)

</div>

----
CVGlue is a computer vision toolkit that integrates face detection, pose estimation, quality assessment, and other tasks. It provides a unified OpenCV/PyTorch data processing interface and supports automated dataset annotation.


## ✨ Core Features
**Out-of-the-box Usage**: Unified interface for multiple algorithms  
**Efficient Visualization**: Single-line Jupyter display for multiple image formats (OpenCV/PIL/Tensor/file paths)  
**Dataset Support**: Seamless integration with [IAP Dataset](https://github.com/Lamply/IAPDataset) annotation pipeline

| Category        | Capabilities                     | Supported Algorithms                   |
|-----------------|----------------------------------|----------------------------------------|
| **Face Analysis** | Detection/Landmarks/Pose Estimation | RetinaFace, AdaptiveWingLoss, FSA-Net  |
| **Quality Assessment** | Face Quality Scoring           | TFace FIQA                             |
| **Data Tools**    | Preprocessing/Visualization/Annotation | OpenCV & PyTorch wrappers            |
| **Extension Modules** | Inpainting/Generic Segmentation | LaMa, SegmentAnything (In Development) |


## 🚀 Quick Example

Create a Jupyter Notebook:

```python
import cv2
import cvglue
from cvglue import displayer as display

parser = cvglue.parser.get_parser('lamply-faceid')
img = cv2.imread('tests/data/images/single_face_img.jpg')

anno = parser.parse_img(img)
iap_data = (img, anno)
img_disp = display.render_lamply(iap_data)
display.show([img, img_disp])
```

Result:
<div align="center"> <img src="docs/output.png" width="600"/> </div>

## ⚙️ Installation

```bash
pip install cvglue
```

Download large models manually to `TORCH_HOME`:
- Resnet50_Final.pth: https://github.com/biubug6/Pytorch_Retinaface
- WFLW_4HG.pth: https://github.com/protossw512/AdaptiveWingLoss
- SDD_FIQA_checkpoints_r50.pth: https://github.com/Tencent/TFace


## 🔌 Third-Party Integrations

| Third-Party Code  | Status | Purpose                   | Original Project                                                                                                         |
| ----------------- | ------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| FaceDetector      | ✅      | Face Detection            | [https://github.com/biubug6/Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark) |
| AdaptiveWing      | ✅      | Facial Landmark Detection | [https://github.com/protossw512/AdaptiveWingLoss](https://github.com/protossw512/AdaptiveWingLoss)                       |
| HeadPoseDetector  | ✅      | Head Pose Estimation      | [https://github.com/shamangary/FSA-Net](https://github.com/shamangary/FSA-Net)                                           |
| TFace             | ✅      | Face Quality Assessment   | [https://github.com/Tencent/TFace](https://github.com/Tencent/TFace)                                                     |
| LaMa              | ⏳      | Image Inpainting          | [https://github.com/advimman/lama](https://github.com/advimman/lama)                                                     |
| SegmentAnything   | ⏳      | Generic Segmentation      | [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)             |
| InsightFace       | ✅      | Face ID/Attributes        | [https://github.com/TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)                         |
| AttributeDetector | ✅      | Face Attribute Analysis   | [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)                             |
