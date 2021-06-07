#Done by Group 20, EE626


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

                    
# load model
model = load_model('gender_detection.h5')

# open webcam
webcam = cv2.VideoCapture(0)
    
label = {0:"female",1:"male"}

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, image = webcam.read()

    # apply face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, 1.1, 7)
    # loop through detected faces
    for x, y, w, h in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150))
        img_scaled = face / 255.0
        reshape = np.reshape(img_scaled, (1, 150, 150, 3))
        img = np.vstack([reshape])
        res = model.predict(img)
        clarity = res[0,0] - res[0,1]
        result = np.argmax(res)

        if result == 0 and clarity > 0.8:
            cv2.rectangle(image, (x - 10, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.rectangle(image, (x - 10, y - 50), (x + w, y), (255, 0, 0), -1)
            cv2.putText(image, label[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        else:
            cv2.rectangle(image, (x - 10, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.rectangle(image, (x - 10, y - 50), (x + w, y), (255, 0, 0), -1)
            cv2.putText(image, label[1], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # display output
    cv2.imshow("gender detection", image)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
