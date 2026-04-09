# CITS4402 Face Detection and Matching Project
## Setup
### 1. Create a virtual environment
```bash
python3 -m venv .venv
```
### 2. Activate the virtual environment
Windows
```bash
.venv\Scripts\activate
```
macOS/Linux
```bash
source .venv/bin/activate
```
### 3. Install project dependencies
```bash
pip install -r requirements.txt
```
### 4. Download the OpenCV DNN face detector model
```bash
mkdir -p models

wget -O models/opencv_face_detector.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -O models/opencv_face_detector.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```
## If the script cannot find the model, check that:
the path is correct
the file exists
the file is not empty
Running the script
```bash
python script.py
```