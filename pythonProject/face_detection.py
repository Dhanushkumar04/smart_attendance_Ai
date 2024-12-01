import cv2
import numpy as np
import face_recognition as face_rec

def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
dhanush = face_rec.load_image_file('images/dhanush.jpg')
dhanush = resize(dhanush,0.50)
dhanush = cv2.cvtColor(dhanush,cv2.COLOR_BGR2RGB)
dhanush2 = face_rec.load_image_file('images/dhanush2.jpg')
dhanush2 = resize(dhanush2,0.50)
dhanush2 = cv2.cvtColor(dhanush2,cv2.COLOR_BGR2RGB)

faceLocation_dhanush = face_rec.face_locations(dhanush)[0]
encode_dhanush = face_rec.face_encodings(dhanush)[0]
cv2.rectangle(dhanush,(faceLocation_dhanush[3],faceLocation_dhanush[0],faceLocation_dhanush[1],faceLocation_dhanush[2]),(255,0,255),3)

faceLocation_dhanush2 = face_rec.face_locations(dhanush2)[0]
encode_dhanush2 = face_rec.face_encodings(dhanush2)[0]
cv2.rectangle(dhanush2,(faceLocation_dhanush[3],faceLocation_dhanush[0],faceLocation_dhanush[1],faceLocation_dhanush[2]),(255,0,255),3)

results = face_rec.compare_faces([encode_dhanush],encode_dhanush2)
print(results)




cv2.imshow("main_img", dhanush)
cv2.imshow("test_img", dhanush2)
cv2.waitKey(0)
cv2.destroyAllWindows()