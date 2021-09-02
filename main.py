import cv2
import numpy as np
import face_recognition
import os
import time


train_path = 'FaceCollection'
images = []
classNames = []
mylist = os.listdir(train_path)


for cl in mylist:
    curImg = cv2.imread(f'{train_path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

#test with dataset
#----------------------------------------------
# test_path = 'FaceTest'
# testImages = []
# testlist = os.listdir(test_path)
#
# for cl in testlist:
#     curImg = cv2.imread(f'{test_path}/{cl}')
#     testImages.append(curImg)
# for img in testImages:
#     facesCurImg = face_recognition.face_locations(img)[0]
#     encodeCurImg = face_recognition.face_encodings(img)[0]
#     matches = face_recognition.compare_faces(encodeListKnown, encodeCurImg)
#     faceDis = face_recognition.face_distance(encodeListKnown, encodeCurImg)
#     print(faceDis)
#     matchIndex = np.argmin(faceDis)
#
#     if matches[matchIndex]:
#         name = classNames[matchIndex].upper()
#         print(name)
#         y1, x2, y2, x1 = facesCurImg
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, name, (x1 + 6, x2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#
#     cv2.imshow('Img', img)
#     cv2.waitKey(1)

#test with camera
#----------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
