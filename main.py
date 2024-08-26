from flask import Flask, render_template, Response
import cv2
import math as m
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize posture detection setup
good_frames = 0
bad_frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

good_posture_shoulder_angles = [10, 12, 8, 9, 11, 13, 10, 9, 12]
good_posture_neck_nose_angles = [5, 6, 4, 5, 7, 6, 5, 4, 5]

shoulder_mean = np.mean(good_posture_shoulder_angles)
shoulder_std = np.std(good_posture_shoulder_angles)

neck_nose_mean = np.mean(good_posture_neck_nose_angles)
neck_nose_std = np.std(good_posture_neck_nose_angles)

SHOULDER_SLOPE_THRESHOLD = shoulder_mean + 2 * shoulder_std
NECK_NOSE_SLOPE_THRESHOLD = neck_nose_mean + 2 * neck_nose_std

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

# Define functions to calculate distance and angle
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.degrees(m.atan2(y2 - y1, x2 - x1))
    return theta

def sendWarning():
    print("Bad posture detected for over 3 minutes!")

# Generate video feed
def generate_frames():
    global good_frames, bad_frames

    while True:
        success, image = cap.read()
        if not success:
            break

        fps = cap.get(cv2.CAP_PROP_FPS)
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if keypoints.pose_landmarks:
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            nose_x = int(lm.landmark[lmPose.NOSE].x * w)
            nose_y = int(lm.landmark[lmPose.NOSE].y * h)
            neck_x = (l_shldr_x + r_shldr_x) // 2
            neck_y = (l_shldr_y + r_shldr_y) // 2

            shoulder_slope = findAngle(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            neck_nose_slope = findAngle(neck_x, neck_y, nose_x, nose_y)
            
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, yellow, -1)
            cv2.circle(image, (nose_x, nose_y), 7, yellow, -1)
            cv2.circle(image, (neck_x, neck_y), 7, yellow, -1)

            angle_text_string = f'Shoulder Slope: {int(shoulder_slope)} | Neck-Nose Slope: {int(neck_nose_slope)}'

            # if abs(shoulder_slope) < SHOULDER_SLOPE_THRESHOLD and abs(neck_nose_slope) < NECK_NOSE_SLOPE_THRESHOLD:
    
            cv2.putText(image, angle_text_string, (10, 30), font, 0.8, light_green, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), green, 4)
            cv2.line(image, (neck_x, neck_y), (nose_x, nose_y), green, 4)
        # else:
            #     good_frames = 0
            #     bad_frames += 1

            #     cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            #     cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), red, 4)
            #     cv2.line(image, (neck_x, neck_y), (nose_x, nose_y), red, 4)
            #     cv2.putText(image, "Incorrect", (10, h - 50), font, 0.9, red, 2)

  

        # Convert image to JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
