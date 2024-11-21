# Field Tracking of Insects Using a Stereoscopic Event-Based Camera Setup

paper:

Video:

This is the official implementation of the **JEB** paper
Field Tracking of Insects Using a Stereoscopic Event-Based
Camera Setup.

## Method Overview

![method.png](method.png)

Existing OpenCV-based methods have been utilized for the detection and tracking of insects, specifically honeybees in this case using a stereoscopic setup of event-based cameras. The methodology involves the sequential application of image processing techniques such as erosion, dilation, and contour detection, followed by the implementation of a Kalman filter to accurately identify and track individual bees within the field of view. This approach leverages well-established and robust algorithms to achieve reliable performance in dynamic and complex environments. The study evaluates the algorithm in two interconnected components: the detection phase and the tracking phase.

![](./assets/results.png)

## Project Overview

1.Clone this repository

```
git clone
```

2.Install the requirements:

```
pip install -r requirements.txt
```
3. Store the calibration and video files recorded from the event cameras in the "Media" folder
```
mkdir Media
```
4. Run the python script
```
python3 3D_event_track.py
```
## Camera Focusing
Video comes here

## Stereo Calibration

We have used [XYZ's repository](https://github.com/username/repository) to calibrate my event-based stereo setup. You can refer to his repository to generate the `calibration.json` file.

video of stereo calibration

## Stereo Matching

In stereo vision, solving the correspondence problem—identifying matching features in left and right images—is crucial for accurate depth estimation and 3D reconstruction. Our approach addresses this challenge using image rectification and epipolar geometry. Rectification aligns stereo images so epipolar lines are horizontal, reducing the correspondence search from two dimensions to one. The OpenCV function `cv2.remap` is used for pixel transformations during rectification.

The fundamental matrix \( F \) computes epipolar lines, which represent possible feature locations in the second image for points detected in the first. Correspondences are determined by minimizing the perpendicular distance between candidate features and their respective epipolar lines. This process is optimized using the Hungarian algorithm to ensure unique matches by minimizing cumulative epipolar distances.

For high-density scenarios, additional parameters—distance, velocity, and size—are incorporated into the cost function to reject false correspondences, ensuring consistency in motion and size. Matches with a cost below a threshold are tracked across frames for reliable object identification. Once correspondences are established, disparity is calculated as the horizontal distance between matched points, enabling triangulation-based depth estimation and robust 3D reconstruction. Features are assigned unique IDs for consistency over time, supporting depth analysis and scene reconstruction.

matching video goes here

## 3D Reconstruction

## Acknowledgments

We thank XYZ for releasing their code to calibrate the stereo cameras.
[XYZ](https://github.com/cpeng93/PDRF)
