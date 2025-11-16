import cv2
import numpy as np
import imutils
from tkinter import *
from tkinter import filedialog

video_path = None   # Stores selected video path


# -------------------------------
# GUI FUNCTIONS
# -------------------------------
def browse_file():
    global video_path
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        status_label.config(text=f"Selected: {video_path}")


def start_detection():
    global video_path
    window.destroy()   # close GUI window
    run_detection(video_path)


def quit_app():
    window.destroy()


# -------------------------------
# MOVING OBJECT DETECTION
# -------------------------------
def run_detection(path):

    # If no file selected → use webcam
    if path is None or path == "":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    # Read 30 frames for background
    frames = []
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)

    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    gray_median = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)

    # Restart video
    cap.release()
    cap = cv2.VideoCapture(path if path else 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background removal
        diff = cv2.absdiff(gray_frame, gray_median)

        # Blur + threshold
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

        # -----------------------------
        # FIX MULTIPLE BOXES PROBLEM
        # -----------------------------

        # Clean noise
        kernel = np.ones((7, 7), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Contours after cleanup
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        boxes = []

        for c in contours:
            if cv2.contourArea(c) < 1500:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        # Merge overlapping boxes
        def merge_boxes(boxes):
            if len(boxes) == 0:
                return []

            boxes = sorted(boxes)
            merged = [boxes[0]]

            for current in boxes[1:]:
                prev = merged[-1]

                if current[0] <= prev[2] and current[2] >= prev[0]:
                    prev[0] = min(prev[0], current[0])
                    prev[1] = min(prev[1], current[1])
                    prev[2] = max(prev[2], current[2])
                    prev[3] = max(prev[3], current[3])
                else:
                    merged.append(current)

            return merged

        merged_boxes = merge_boxes(boxes)
        object_count = len(merged_boxes)

        # Draw merged boxes
        for (x1, y1, x2, y2) in merged_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 3)

        # Project Title
        cv2.putText(frame, "Moving Object Detection - Project by Shruti",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Object Count
        cv2.putText(frame, f"Objects Detected: {object_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

        cv2.imshow("Personalized Moving Object Detector", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# TKINTER GUI WINDOW
# -------------------------------
window = Tk()
window.title("Moving Object Detection - Shruti")
window.geometry("450x200")
window.configure(bg="#E8E8E8")

Label(window, text="Moving Object Detection - Project by Shruti",
      font=("Arial", 14, "bold"), bg="#E8E8E8").pack(pady=10)

Button(window, text="Browse Video", command=browse_file,
       font=("Arial", 12), width=20, bg="#6c5ce7", fg="white").pack(pady=5)

Button(window, text="Start Detection", command=start_detection,
       font=("Arial", 12), width=20, bg="#00b894", fg="white").pack(pady=5)

Button(window, text="Quit", command=quit_app,
       font=("Arial", 12), width=20, bg="#d63031", fg="white").pack(pady=5)

status_label = Label(window, text="No video selected (Webcam will be used)",
                     bg="#E8E8E8", fg="black", font=("Arial", 10))
status_label.pack()

window.mainloop()
