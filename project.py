import mediapipe.python.solutions.hands as hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import cv2 as cv


class face:
    def __init__(self,filename:str):
        self.filename = filename

        self.face_cascade = cv.CascadeClassifier(self.filename)
        self.mp_hands = hands
        self.hands_detector = self.mp_hands.Hands()
        self.cap = cv.VideoCapture(0)

    def looper(self):

        while True:
            is_true, frame = self.cap.read()
            if not is_true:
                break

            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            img2rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.hands_detector.process(img2rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        print(id, landmark)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            face_list = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in face_list:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv.imshow("Frame", frame)
            if cv.waitKey(1) & 0xFF == ord('x'):
                break


        self.cap.release()
        cv.destroyAllWindows()


ob = face(r"face.xml")
ob.looper()
