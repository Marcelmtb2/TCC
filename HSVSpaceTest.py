# import cv2
from __future__ import print_function
import cv2 as cv
import argparse
 
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
 
 
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
 
 
 
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
 
 
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
 
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
 
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
 
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
 
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
 
pasta = "video\\videosMock\\"

# cap = cv2.VideoCapture(r"video\video.mp4")
# cap = cv2.VideoCapture(r"video\teste.mp4")

objeto1 = "cabo luz off.mp4"
objeto2 = "cabo movimento maos luz on.mp4"
objeto3 = "caixa clara movimento maos luz on.mp4"
objeto4 = "caixa desde inicio luz on.mp4"
objeto5 = "caixa luz off.mp4"
objeto6 = "caixa mudanca iluminacao.mp4"
objeto7 = "Paquimetro luz off.mp4"
objeto8 = "Paquimetro mao presente luz off.mp4"
objeto9 = "Paquimetro para caixa luz off.mp4"
objeto10 = "Regua luz off.mp4"
objeto11 = "regua refletiva luz on.mp4"

objeto12 = "BaixaIluminacao100luxSombraForte.mp4"
objeto13 = "TrocaObjetosAutofocoAtrapalha.mp4"
objeto14 = "Iluminacao800_560lux.mp4"
objeto15 = "Objeto15segs.mp4"
objeto16 = "Objeto15segSubstituido.mp4"
objeto17 = "objeto3segs.mp4"
objeto18 = "ObjetoInicio.mp4"
objeto19 = "ObjetoOrtogonalDiagonal.mp4"
objeto20 = "ObjetoReposicionado.mp4"
objeto21 = "OclusãoMão.mp4"
objeto22 = "OclusãoTempMão.mp4"
objeto23 = "TemperaturaCor3k_9k.mp4"  # - para gerar valores HSV do background
objeto24 = "ContrasteTemperaturaCor3k_9k.mp4"  # - valores HSV do background

objeto = objeto14

cap = cv.VideoCapture(pasta + objeto)
#cap = cv.VideoCapture(args.camera)
 

 
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
 
 
 
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
 
 
while True:
    
    ret, frame = cap.read()
    if frame is None:
        break

    frame = cv.resize(frame, None, fx=0.25, fy=0.25,
                         interpolation=cv.INTER_AREA)
 
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
 
    
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    
 
    key = cv.waitKey(230)
    if key == ord('q') or key == 27:
        break



# Callback function for trackbars (dummy function for now)
# def nothing(x):
#     pass

# # Load the video
# # video_path = "example.mp4"qq

# # Check if the video file opened successfully
# if not cap.isOpened():
#     print("Error: Cannot open video file.")
#     exit()

# # Create a named window for the video and trackbars
# cv2.namedWindow("Video with Trackbars", cv2.WINDOW_NORMAL)

# # Create trackbars
# cv2.createTrackbar("Min Hue", "Video with Trackbars", 0, 179, nothing)
# cv2.createTrackbar("Max Hue", "Video with Trackbars", 0, 179, nothing)
# cv2.createTrackbar("Min Saturation", "Video with Trackbars", 0, 255, nothing)
# cv2.createTrackbar("Max Saturation", "Video with Trackbars", 0, 255, nothing)

# while True:
#     # Read the next frame from the video
#     ret, frame = cap.read()

#     # If the video ends, rewind to the beginning
#     if not ret:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to the first frame
#         continue

#     # Resize the frame (optional, adjust as needed)
#     scale_percent = 75  # Scale down by 75%
#     width = int(frame.shape[1] * scale_percent / 100)
#     height = int(frame.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

#     # Display the frame
#     cv2.imshow("Video with Trackbars", resized_frame)

#     # Handle keypress (e.g., ESC to exit)
#     key = cv2.waitKey(30) & 0xFF
#     if key == 27:  # Exit on ESC key
#         break

# # Release video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
