import preprocessing
from sklearn.externals import joblib

print("dsadas")
mlp = joblib.load("classifier.joblib") #load model

img = preprocessing.read_image("path to image") #read the image we want to classify

img = img/255 #divide image values by 255 to shrink them between 0 and 1

img = img.reshape((1, len(img))) #reshape the image values to shape n x 2500, where n is number of samples

prediction = mlp.predict(img) #use model to classify the image

print(prediction[0])