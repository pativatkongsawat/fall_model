import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


model = load_model('lstm_fall_detection.h5')

SELECTED_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
sequence_length = 12


mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)


def detectPose(image, pose):
    imgHeight, imgWidth, _ = image.shape
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    
    landmarks = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if idx in SELECTED_LANDMARKS:
                landmarks.append((landmark.x * imgWidth, landmark.y * imgHeight))
    return results, landmarks


def predict_fall(model, sequence):
    sequence = np.array(sequence)
    sequence = sequence.reshape(1, sequence_length, len(SELECTED_LANDMARKS) * 2)
    prediction = model.predict(sequence)
    return prediction[0][0] > 0.5  #


def draw_landmarks(image, results):
    if results.pose_landmarks:
        # วาดจุดทั้งหมด
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv.circle(image, (x, y), 5, (0, 255, 0), -1)  
        
        
        connections = mpPose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = (int(results.pose_landmarks.landmark[start_idx].x * image.shape[1]),
                          int(results.pose_landmarks.landmark[start_idx].y * image.shape[0]))
            end_point = (int(results.pose_landmarks.landmark[end_idx].x * image.shape[1]),
                        int(results.pose_landmarks.landmark[end_idx].y * image.shape[0]))
            cv.line(image, start_point, end_point, (255, 0, 0), 2)


def process_video(video_path, model, output_path='output1_video.mp4'):
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    
   
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    sequence = []
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        results, landmarks = detectPose(frame, pose)
        
        if landmarks:
            
            draw_landmarks(frame, results)
            
           
            flattened_landmarks = [coord for landmark in landmarks for coord in landmark]
            sequence.append(flattened_landmarks)
            
            
            if len(sequence) >= sequence_length:
                is_fall = predict_fall(model, sequence[-sequence_length:])
                if is_fall:
                    cv.putText(frame, "FALL", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv.putText(frame, "ADL", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        cv.imshow("Fall Detection", frame)
        
        
        out.write(frame)
        
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
   
    cap.release()
    out.release()
    cv.destroyAllWindows()


video_path = "485222229_28833105016305079_8107917099308813204_n.mp4"  
output_path = "output1_video.mp4"  
process_video(video_path, model, output_path)