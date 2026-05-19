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
### 4. Download the face detector and landmark model files
```bash
# If accessing this from the submitted zip file, models + all the files required inside should be already present. If accessing from Github, proceed with the commands below.
# If having difficulties running the commands, download the file from the link and move it into models/
mkdir -p models

# OpenCV DNN face detector files
wget -O models/opencv_face_detector.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

wget -O models/opencv_face_detector.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# dlib 5-point facial landmark model
wget -O models/shape_predictor_5_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

bzip2 -dk models/shape_predictor_5_face_landmarks.dat.bz2

# dlib face recognition ResNet model
# Windows (PowerShell):
curl -o models/dlib_face_recognition_resnet_model_v1.dat.bz2 \
  http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
python -c "import bz2, shutil; shutil.copyfileobj(bz2.open('models/dlib_face_recognition_resnet_model_v1.dat.bz2', 'rb'), open('models/dlib_face_recognition_resnet_model_v1.dat', 'wb'))"

# macOS/Linux:
wget -O models/dlib_face_recognition_resnet_model_v1.dat.bz2 \
  http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bzip2 -dk models/dlib_face_recognition_resnet_model_v1.dat.bz2
```
## Notes
- The `models/` folder is ignored by Git, so the model files must be downloaded locally.
- The required model files must be saved at:
  - `models/opencv_face_detector.prototxt`
  - `models/opencv_face_detector.caffemodel`
  - `models/shape_predictor_5_face_landmarks.dat`
  - `models/dlib_face_recognition_resnet_model_v1.dat`
## If the script cannot find the model files, check that:
- the paths are correct
- all files exist
- none of the files are empty
- the 2 `.bz2` files has been extracted so that `shape_predictor_5_face_landmarks.dat` and `dlib_face_recognition_resnet_model_v1.dat` are present
- To control the strictness in clustering faces (DBSCAN), change the EPS value (Max Euclidean Distance) in line 551 of script. Larger number for less clusters, looser clusters and smaller number for more clusters, stricter clusters
## Running the script
```bash
python script.py
```