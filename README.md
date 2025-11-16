🚨 Moving Object Detection

A simple and efficient computer vision project that detects moving objects in video streams using frame differencing and contour analysis. This system continuously compares consecutive frames to identify motion and highlight moving regions in real time.

🔍 Features
Detects moving objects using background subtraction / frame differencing
Real-time video processing with OpenCV
Highlights detected objects with bounding boxes
Works with webcam or pre-recorded videos

🛠️ Tech Stack
Python
OpenCV
NumPy

▶️ How It Works
Capture frames from video or webcam
Convert to grayscale & apply Gaussian blur
Compare consecutive frames
Detect motion through thresholding and contours
Draw bounding boxes around moving objects
