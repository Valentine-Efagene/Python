import cv2
import numpy as np
import os

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('data/xml/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if(len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('data/xml/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if(len(faces) == 0):
        return None, None

    return faces

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    
    for dir_name in dirs:
    
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
    
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
    
            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return faces, labels

data_folder_path = "data/images"
faces, labels = prepare_training_data(data_folder_path)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

subjects = ["Philip", "Valentyne", "great", "Chidinma"]
img = cv2.imread("data/images/test/test2.jpg")
img = cv2.resize(img, (960, 540))  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detect_faces(img)

for face in faces:
    (x, y, w, h) = face
    face, rect = gray[y:y+w, x:x+h], face
    label = face_recognizer.predict(face)[0]
    label_text = subjects[label]
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, label_text, (rect[0], rect[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()