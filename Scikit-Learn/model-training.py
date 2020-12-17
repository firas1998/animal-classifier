import preprocessing

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

#save training and testing data as numpy arrays
preprocessing.save_images("path to dataset", "train_images", "train_labels", "test_images", "test_labels")

train_data, train_labels = preprocessing.load_training_data("train_images", "train_labels") #load training data
test_data, test_labels = preprocessing.load_testing_data("test_images", "test_labels")#load testing data
print("training")
#create an ANN with one hidden layer that contains 500 neurons, with an activation of relu for all neurons in the hidden layers, that trains for 50 epochs using the adam oprimizer
mlp = MLPClassifier(hidden_layer_sizes=(500), activation="relu", max_iter=50,
                    solver='adam', verbose=1, batch_size=50)

mlp.fit(train_data, train_labels) #train model on the training data

score = mlp.score(test_data, test_labels) #evaluate the model using the test data
print("Test set score: %f" % score)

joblib.dump(mlp, "classifier.joblib") #save the model