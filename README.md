# Virtual-Make-Up

Project I worked on using Dlib and a convolutional network to segment the lips and the face of a person and then perform a color swap. The entire project was done in Google Colab inside my google drive.

I've specified what versions open keras, numpy, opencv, etc. I'm using.

```
These are the two image I used as my test images. I've included them in this github page
```
<img src="girl-no-makeup.jpg" width="324" height="324"><img src="girl-no-makeup-2.jpg" width="324" height="324">


## Lip Color Swap

Lip color swap with Dlib 68 point landmark detector!\
Results:\
<img src="result/girl-no-makeup-lip-color-change.png" width="324" height="324"><img src="result/girl-no-makeup-2-lip-color-change.png" width="324" height="324">

- Requirements(version I used):
  - OpenCV(4.1.2)
  - Numpy(1.18.5)
  - Dlib(19.18.0)
  - Matplotlib(3.2.2)
  - ![shape_predictor_68_face_landmarks.dat](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)
    - The file is about 95mb, I couldn't upload it to Github, the link is from someone else's github. Or you can download it somewhere else

## How it works

<img src="https://www.researchgate.net/publication/327500528/figure/fig9/AS:668192443748352@1536320901358/The-ibug-68-facial-landmark-points-mark-up.ppm" width="324" height="324">

### Obtaining the facial landmarks

The first step is to use facial landmark detection from the image of the face. Using the 68 face predictor model settings, the ```face detector``` and ```landmark detector``` is loaded from dlib.

```
# Landmark model location
PREDICTOR_PATH =  "shape_predictor_68_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
```

Using ```getLandmarks``` from ```faceblendcommon```, the 68 landmark points are obtained from the face.
```
points = fbc.getLandmarks(faceDetector, landmarkDetector, img)
```
<img src="results/landmark_img.PNG" width="324" height="324">


## Hair Color Swap

- Requirements:
  - ![Hair segmentation model](https://github.com/thangtran480/hair-segmentation/releases)
  - Keras(2.4.3)
  
### Helper method
```
def predict(image, height=224, width=224):
    im = image.copy()
    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)
    
    pred = model.predict(im)
    mask = pred.copy()
    mask = mask.reshape((224, 224,1))
    row, col, _ = image.shape
    mask = cv2.resize(mask, (col, row))
    return mask
```
