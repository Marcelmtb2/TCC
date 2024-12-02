import cv2

cap = cv2.VideoCapture(r"video\video.mp4")
orb = cv2.ORB_create()
ret, template = cap.read()
keypoints_template, descriptors_template = orb.detectAndCompute(template, None)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    keypoints_frame, descriptors_frame = orb.detectAndCompute(frame, None)

    # Matcher para encontrar correspondências entre as características
    # Desenhar os pontos de interesse na imagem
    img_with_keypoints = cv2.drawKeypoints(frame, keypoints_frame, None, color=(0, 255, 0), flags=0)
    print(descriptors_template)
    print(descriptors_frame)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_template, descriptors_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    # Desenhar as correspondências
    match_img = cv2.drawMatches(template, keypoints_template, frame, keypoints_frame, matches[:10], None, flags=2)

    cv2.imshow('Feature Matching', match_img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
