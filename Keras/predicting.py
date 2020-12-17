import preprocessing
from keras.models import load_model

model = load_model("model.h5") #load CNN model

img = preprocessing.read_image("path to image") #read the image we want to classify

img_for_prediction = img.astype("float")/255 #convert image values to float, and shrink them between 0 and 1

img_for_prediction = img_for_prediction.reshape((1,50,50,3)) #reshape multidimentional array to have the shape 1x50x50x3, 1 being the number of samples

prediction = model.predict(img_for_prediction) #use model to predict the class

prediction = prediction[0].argmax() #get the class with the highest probability

print(prediction)