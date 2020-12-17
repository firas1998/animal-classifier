from tensorflow.python.keras import layers, models, utils
import preprocessing

model = models.Sequential() #create sequential model

model.add(layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu', input_shape=(50, 50, 3))) #add convolutional layer with 32 kernels each of size 3x3 and relu activation
model.add(layers.MaxPool2D(pool_size=2)) #add max pooling layer with pooling of size 2x2

model.add(layers.Conv2D(filters=32, kernel_size=3, padding="same", activation='relu')) #add convolutional layer with 32 kernels each of size 3x3 and relu activation
model.add(layers.MaxPooling2D(pool_size=2)) #add max pooling layer with pooling of size 2x2

model.add(layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")) #add convolutional layer with 64 kernels each of size 3x3 and relu activation
model.add(layers.MaxPooling2D(pool_size=2)) #add max pooling layer with pooling of size 2x2

model.add(layers.Dropout(0.5)) #add dropout layer with 0.5 dropout rate
model.add(layers.Flatten()) #flatten the output of the max pooling layer
model.add(layers.Dense(1028, activation="relu")) #add a fully connected layer with 1028 neurons and relu activation
model.add(layers.Dropout(0.5)) #add dropout layer with 0.5 dropout rate

model.add(layers.Dense(7, activation="softmax")) #add a fully connected layer with 7 neurons and softmax activation, this will be the output laayer
model.summary() #print model summary

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compile model with categorical cross entropy loss function, and the adam optimizer

#save training and testing data as numpy arrays
preprocessing.save_images("path to dataset", "train_images", "train_labels", "test_images", "test_labels")

train_data, train_labels = preprocessing.load_training_data("train_images", "train_labels") #load training data
test_data, test_labels = preprocessing.load_testing_data("test_images", "test_labels") #load testing data

train_labels = utils.to_categorical(train_labels) #convert training labels to one-hot encoded labels
test_labels = utils.to_categorical(test_labels) #convert testing labels to one-hot encoded labels

#train model for 30 epochs, updating the weights after every 50 samples
model.fit(train_data, train_labels, batch_size=50,validation_data=(test_data, test_labels), epochs=30, verbose=1)

score = model.evaluate(test_data, test_labels, verbose=1) #evaluate model with the testing data
print(score)

model.save("model.h5")