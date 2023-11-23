# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
modeule=tf.keras.models.load_model('keras_model.h5')
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img=cv2.resize(frame,(224,224))
    testingimages=np.array(img,dtype=np.float32)
    testingimages=np.expand_dims(testingimages,axis=0)
    testingimages=testingimages/255.0
    preditedOutput=modeule.predict(testingimages)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()