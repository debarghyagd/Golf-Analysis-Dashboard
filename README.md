# Golf Play Analytics Dashboard

<video width="320" height="240" controls>
  <source src="https://github.com/debarghyagd/Golf-Analysis-Dashboard/blob/main/streamlit-driver-2023-09-07-18-09-53.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


Welcome to the Golf Play Analytics Dashboard, a tool for golf enthusiasts. This dashboard is built using Streamlit and offers a range of features to help you analyze golf ball trajectories and identify golfer postures using the TFLite Movenet Thunder model.

## Features

### 1. Predicting and Visualizing Golf Ball Trajectory

With this feature, you can:

- **Input Swing Data**: Enter essential details about the golf swing, such as club selection, swing speed, and angle of attack.
- **Visualize Trajectory**: The algorithm will calculate and predict the trajectory of the golf ball based on your input data, while accounting for the applied spin/Magnus effect and Air Resistance. The trajectory is displayed graphically, making it easy to understand and analyze.

### 2. Identifying Golfer Posture using TFLite Movenet Thunder Computer Vision Model

This feature allows you to:

- **Upload Video Footage**: Simply upload, link, capture, or choose form the local dataase, a video of a golfer's swing.
- **Golfer Posture Analysis**: Our integrated TFLite Movenet Thunder CV model will analyze the golfer's posture throughout the swing.
- **Visual Feedback**: Receive visual feedback on key posture elements, such as body alignment, and posture angles.

## Dependencies

- Python 3.x
- Streamlit
- OpenCV
- TFLite Movenet Thunder
- Matplotlib
- NumPy

## Run
```
cd ./'Golf Dashboard'
streamlit run driver.py
```
