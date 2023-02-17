#Analyze one image file at a time
import tensorflow.keras
import numpy as np
import cv2

#disable scientific notation
np.set_printoptions(suppress=False)
#load the model
model = tensorflow.keras.models.load_model('derma_model_110721.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#test image
img = cv2.imread('../Basal Cell Carcinoma/bbc2a.jpg')
#img = cv2.imread('../Basal Cell Carcinoma/bcc_test_set/bbc_test1.jpg')
#img = cv2.imread('../Squamous Cell Carcinoma/scc_test_set/scc_test3.jpg')
#img = cv2.imread('../Malignant Melanoma/mm_test_set/mm_test1.jpg')
#img = cv2.imread('../Benign Mole/bm_test_set/bm_test2.jpg')
img = cv2.resize(img,(224, 224))
#img = cv2.resize(img,(224, 224),fx=0,fy=0, interpolation=cv2.INTER_CUBIC)

image_array = np.asarray(img)

#font config
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

#Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

prediction = model.predict(data)

#print(prediction)
for i in prediction:
    if i[0] > 0.4:
        text = "Basal Cell Carcinoma"
    if i[1] > 0.4:
        text = "Squamous Cell Carcinoma"
    if i[2] > 0.4:
        text = "Malignant Melanoma"
    if i[3] > 0.4:
        text = "Benign Mole"
    #print("Basal Cell Carcinoma = %.4f, Squamous Cell Carcinoma = %.4f, Malignant Melanoma = %.4f" %(i[0]*100, i[1]*100, i[2]*100))
    print("==============================")
    print("| Skin Analysis Result       |")
    print("==============================")
    print("Basal Cell Carcinoma = "+"{:.4%}".format(i[0]))
    print("Squamous Cell Carcinoma = "+"{:.4%}".format(i[1]))
    print("Malignant Melanoma = "+"{:.4%}".format(i[2]))
    print("Benign Mole = "+"{:.4%}".format(i[3]))
    img = cv2.resize(img,(500, 500))
    #cv2.putText(img,text,(10,30),cv2.FONT_ITALIC,1,(0,255,0),2)
    cv2.putText(img,text,org,font,fontScale,color,thickness,cv2.LINE_AA)
cv2.imshow('img',img)
cv2.waitKey(0)

