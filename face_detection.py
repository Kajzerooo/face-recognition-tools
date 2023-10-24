import cv2

# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera, you can change this to specify a different camera.

while True:
    # Odczytaj klatkę z kamery
    ret, frame = cap.read()

    #Flip kamery
    frame = cv2.flip(frame, 1)

    # Konwertuj klatkę w skalę szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrywaj twarze w klatce
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    
    # Rysuj prostokąty wokół twarzy
    #for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #Blur twarzy
    for(x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (0, 0), 15)
        #face = cv2.flip(frame, 1)
        frame[y:y+face.shape[0], x:x+face.shape[1]] = face


   
    # Wyświetl klatkę z wykrytymi twarzami/twarzą
    cv2.imshow('Face Detection + Blur', frame)

    # Zamknij program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zamknij kamerę i wszystkie okna cv2
cap.release()
cv2.destroyAllWindows()
