#Analyze video image at real time
import tensorflow.keras
import numpy as np
import cv2

#disable scientific notation
np.set_printoptions(suppress=True)

#load the model
model = tensorflow.keras.models.load_model('derma_model_110721.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#0 for laptop webcam
cam = cv2.VideoCapture(1)

text = ""

#font config
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

while True:
    _,img = cam.read()
    #img = cv2.resize(img,(448, 448))
    img = cv2.resize(img,(224, 224))
    
 
    #turn the image into a numpy array
    image_array = np.asarray(img)

    #normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    #load the image into the array
    data[0] = normalized_image_array

    #run inference
    prediction = model.predict(data)
    for i in prediction:
        if i[0] > 0.002:
            text = "Basal Cell Carcinoma"
        if i[1] > 0.80:
            text = "Squamous Cell Carcinoma"
        if i[2] > 0.99:
            text = "Malignant Melanoma"
        if i[3] > 0.05:
            text = "Benign Mole"
        
        img = cv2.resize(img,(650, 500))
                
        #original orientation
        #cv2.imshow('img',img)

        #mirror orientation    
        img_mirror = cv2.flip(img, 1)
    
        #overlay text
        #cv2.putText(img,text,org,font,fontScale,color,thickness,cv2.LINE_AA)
        #cv2.putText(img,text,(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),1)
        cv2.putText(img_mirror,text,org,font,fontScale,color,thickness,cv2.LINE_AA)
    
    #display on mirror video on screen
    cv2.imshow('flipped video', img_mirror)
    
    #stop the while-loop by key-in letter 'q'
    if cv2.waitKey(1) & 0xff == ord('q'):
        #display prediction result
        print("==============================")
        print("| Skin Analysis Result       |")
        print("==============================")
        print("Basal Cell Carcinoma = "+"{:.4%}".format(i[0]))
        print("Squamous Cell Carcinoma = "+"{:.4%}".format(i[1]))
        print("Malignant Melanoma = "+"{:.4%}".format(i[2]))
        print("Benign Mole = "+"{:.4%}".format(i[3]))
        break
    
cam.release()
cv2.destroyAllWindows()


