# _Pose estimation on videos using a Neural Network and OpenCV to detect boxing jabs_
<img src="https://github.com/hempelc/jab_detector_NN_pose_estimation/blob/main/image.png" alt="jab" width="400"/>

This notebook estimates poses of people in videos by detecting joints using a Neural Network and connecting the joints. The joints and their connections are then used to predict if the person is doing a boxing jab. 

Therefore, we load a pre-trained Neural Network and apply it to a video or live through a webcam.

The goal of this project is that our program tells us if it detects a boxing jab.

The Neural Network and code to run the pose estimation is taken from https://github.com/quanhua92/human-pose-estimation-opencv

Note: ***this project is not finished***! The video detection of joints and limbs worked poorly - for successful jab detection, the NN might need to be updated. We stopped at that point.


# 1. Prerequisites

Load the OpenCV module


```python
import cv2 as cv
```

Load the pre-trained Neural Network


```python
NN = cv.dnn.readNetFromTensorflow("graph_opt.pb")
```

Set some default parameters


```python
# Resize input to specific width
inwidth = 368
# Resize input to specific height
inheight = 368
# Threshold value for pose parts heat map
thr = 0.2

# Body part detection and pairing
## Adopted from: https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
```

# 2. Apply pose estimation to video


Set the video to use


```python
video = 'The_Boxing_Jab_with_Jeff_Mayweather_Trim.mp4'
```

Open the video and graph pose estimation on video


```python
jab = cv.VideoCapture(video)

while cv.waitKey(1)<0:
    hasFrame,frame = jab.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    NN.setInput(cv.dnn.blobFromImage(frame, 1.0, (inwidth, inheight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = NN.forward()

    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = NN.getPerfProfile()
    freq = cv.getTickFrequency() / 100
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('JabDetector', frame)
```

# 3. Apply pose estimation live through a webcam
### Note: the code is written for a Mac webcam


```python
jab = cv.VideoCapture(0)

jab.set(cv.CAP_PROP_FPS,1000)

while cv.waitKey(1)<0:
    hasFrame,frame = jab.read()

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    NN.setInput(cv.dnn.blobFromImage(frame, 1.0, (inwidth, inheight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = NN.forward()

    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = NN.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('JabTracker', frame)
```

# 4. Do some magic to detect boxing jabs that yet needs to be figured out

Got stuck here
