#pip install mediapipe opencv-python
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_all_coords(landmark):
    print("HELLO")
    #left side coords
    colNames = ["Landmark", "x", "y"]
    df = pd.DataFrame(columns = colNames)
    l_shoulder = {"Landmark" : 'LEFT_SHOULDER', "x" : landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, "y" : landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y}
    l_elbow = {"Landmark" : 'LEFT_ELBOW', "x" : landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, "y" : landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y}
    l_wrist = {"Landmark" : 'LEFT_WRIST',"x" : landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,"y" : landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y}
    l_hip = {"Landmark" : 'LEFT_HIP', "x" : landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,"y" : landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y}
    l_knee = {"Landmark" : 'LEFT_KNEE',"x" :landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,"y" : landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y}
    l_ankle = {"Landmark" : 'LEFT_ANKLE',"x" :landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,"y" : landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y}
    #right side coords    
    r_shoulder = {"Landmark" : 'RIGHT_SHOULDER', "x": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, "y" : landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y}
    r_elbow = {"Landmark" : 'RIGHT_ELBOW', "x": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, "y" : landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y}
    r_wrist = {"Landmark" : 'RIGHT_WRIST', "x" : landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, "y": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y}
    r_hip = {"Landmark" :'RIGHT_HIP',"x" : landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, "y":landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y}
    r_knee = {"Landmark" : 'RIGHT_KNEE', "x" : landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, "y": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y}
    r_ankle = {"Landmark" :'RIGHT_ANKLE',"x" :landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,"y":landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y}
    df = df._append((l_shoulder), ignore_index=True)
    df = df._append((l_elbow), ignore_index=True)
    df = df._append((l_wrist), ignore_index=True)
    df = df._append((l_hip), ignore_index=True)
    df = df._append((l_knee), ignore_index=True)
    df = df._append((l_ankle), ignore_index=True)
    df = df._append((r_shoulder), ignore_index=True)
    df = df._append((r_elbow), ignore_index=True)
    df = df._append((r_wrist), ignore_index=True)
    df = df._append((r_hip), ignore_index=True)
    df = df._append((r_knee), ignore_index=True)
    df = df._append((r_ankle), ignore_index=True)
    print(df)
    return df

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# def get_all_angles(coords):
#     #   elbow angle, armpit angle, hip angle, knee angle
#     # 0 - l_shoulder
#     # 1 - l_elbow
#     # 2 - l_wrist
#     # 3 - l_hip
#     # 4 - l_knee
#     # 5 - l_ankle
#     # 6 - rshoulder
#     # 7 - relbow
#     # 8 - rwrist
#     # 9 - r_hip
#     # 10 - rkne
#     # 11 - r_ankle
#     l_elbow_angle = calculate_angle([coords[0][1], coords[1][2]], [coords[1][1], coords[1][2]], [coords[2][1], coords[2][2]])
#     r_elbow_angle = calculate_angle([coords[6][1], coords[6][2]], [coords[7][1], coords[7][2]], [coords[8][1], coords[8][2]])
    
#     # l_armpit = calculate_angle([coords[6][1], coords[6][2]], [coords[7][1], coords[7][2]], [coords[8][1], coords[8][2]])
#     # r_armpit = 

    # r_hip_angle = 
    # l_hip_angle = 

    # l_knee = 
    # r_knee = 
    

    

cap = cv2.VideoCapture(0)

start_time = int(round(time.time() * 1000))

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark                       
            coords = get_all_coords(landmarks)

            #print(str(round(coords[0][1], 2)))
            for i in range(len(coords)):
                cv2.putText(image, str(round(coords.at[i, "x"], 2)) + ", " + str(round(coords.at[i, "y"], 2)), 
                            tuple(np.multiply([coords.at[i, "x"], coords.at[i, "y"]], [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )    
            
            #angles = get_all_angles(coords)
                
        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()