import mediapipe.python.solutions.hands as hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import cv2 as cv

face_cascade = cv.CascadeClassifier(r"face.xml")

mp_hands = hands

hands_detector = mp_hands.Hands()

cap = cv.VideoCapture(0)

while True:
    is_true, frame = cap.read()
    if not is_true:
        break

    # Convert the frame to grayscale
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Convert the frame to RGB
    img2rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands_detector.process(img2rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            for id, landmark in enumerate(hand_landmarks.landmark):
                
                print(id, landmark)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Detect faces in the grayscale image
    face_list = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in face_list:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv.imshow("Frame", frame)

    # Break the loop on 'x' key press
    if cv.waitKey(1) & 0xFF == ord('x'):
        break

# Release the capture and close the windows
cap.release()
cv.destroyAllWindows()
