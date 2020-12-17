import sys
from PIL import Image
import numpy as np
import os
import cv2


def get_least(classes):
    i = sys.maxsize

    for animal in classes:
        if len(animal) < i:
            i = len(animal)
    return i


def shuffle(animals, labels):
    s = np.arange(animals.shape[0])
    np.random.shuffle(s)
    animals = animals[s]
    labels = labels[s]
    return animals, labels


def read_image(image_path):
    """
    function that reads an image, resizes it, converts it to numpy array, then returns the array

    :param image_path: path to the image to read
    :return: numpy array of the image values
    """

    img = cv2.imread(image_path)  # read image with OpenCV
    img_arr = Image.fromarray(img)
    img_resized = img_arr.resize((50, 50))  # resize image
    img = np.array(img_resized)  # convert image to numpy array

    return img


def save_images(DATASET, training_data_name, training_labels_name, testing_data_name, testing_labels_name):
    """

    :param DATASET: path to dataset
    :param training_data_name: path to save training images
    :param training_labels_name: path to save training labels
    :param testing_data_name: path to save testing images
    :param testing_labels_name: path to save testing labels

    """
    data = []
    labels = []

    pics = []

    #list content of the directories of animals we want to classify
    CATS_DIR = DATASET + "/Cat/"
    cats = os.listdir(CATS_DIR)

    CHICKENS_DIR = DATASET + "/Chicken/"
    chickens = os.listdir(CHICKENS_DIR)

    DOGS_DIR = DATASET + "/Dog/"
    dogs = os.listdir(DOGS_DIR)

    SQUIRRELS_DIR = DATASET + "/Squirrel/"
    squirrels = os.listdir(SQUIRRELS_DIR)

    SHEEP_DIR = DATASET + "/Sheep/"
    sheep = os.listdir(SHEEP_DIR)

    ELEPHANTS_DIR = DATASET + "/Elephant/"
    elephants = os.listdir(ELEPHANTS_DIR)

    HORSES_DIR = DATASET + "/Horse/"
    horses = os.listdir(HORSES_DIR)

    #append all the images in an array
    pics.append(cats)
    pics.append(dogs)
    pics.append(horses)
    pics.append(elephants)
    pics.append(squirrels)
    pics.append(sheep)
    pics.append(chickens)

    #find number of images we will need for each animal/class
    sizeNeeded = get_least(pics)


    #read the number of images needed for each class and label the images
    j = 0

    for cat in cats:
        if j >= sizeNeeded:
            break
        img = read_image(CATS_DIR + cat)
        data.append(img)
        labels.append(0)
        j = j + 1
    j = 0

    for chicken in chickens:
        if j >= sizeNeeded:
            break
        img = read_image(CHICKENS_DIR + chicken)
        data.append(img)
        labels.append(1)
        j = j + 1

    j = 0

    for dog in dogs:
        if j >= sizeNeeded:
            break
        img = read_image(DOGS_DIR + dog)
        data.append(img)
        labels.append(2)
        j = j + 1

    j = 0

    for squirrel in squirrels:
        if j >= sizeNeeded:
            break
        img = read_image(SQUIRRELS_DIR + squirrel)
        data.append(img)
        labels.append(3)
        j = j + 1

    j = 0

    for shep in sheep:
        if j >= sizeNeeded:
            break
        img = read_image(SHEEP_DIR + shep)
        data.append(img)
        labels.append(4)
        j = j + 1

    j = 0

    for elephant in elephants:
        if j >= sizeNeeded:
            break
        img = read_image(ELEPHANTS_DIR + elephant)
        data.append(img)
        labels.append(5)
        j = j + 1

    j = 0

    for horse in horses:
        if j >= sizeNeeded:
            break
        img = read_image(HORSES_DIR + horse)
        data.append(img)
        labels.append(6)
        j = j + 1


    animals = np.array(data)
    labels = np.array(labels)

    #shuffle the training images and labels arrays in the same form
    animals, labels = shuffle(animals, labels)

    data_length = len(animals)

    #seperate training and testing data
    (x_train, x_test) = animals[(int)(0.1 * data_length):] \
        , animals[:(int)(0.1 * data_length)]
    (y_train, y_test) = labels[(int)(0.1 * data_length):] \
        , labels[:(int)(0.1 * data_length)]

    #convert all image values to float and divide them by 255 to make the values between 0 and 1
    x_train = x_train.astype("float") / 255
    x_test = x_test.astype("float") / 255

    #save the training and testing data as numpy arrays
    np.save(training_data_name + ".npy", x_train)
    np.save(training_labels_name + ".npy", y_train)

    np.save(testing_data_name + ".npy", x_test)
    np.save(testing_labels_name + ".npy", y_test)


def load_training_data(training_data_name, training_labels_name):
    """

    :param training_data_name: path to training images
    :param training_labels_name: path to training labels
    :return: training data
    """
    imgs = np.load(training_data_name + ".npy")
    labels = np.load(training_labels_name + ".npy")
    return imgs, labels


def load_testing_data(testing_data_name, testing_labels_name):
    """

    :param testing_data_name: path to testing images
    :param testing_labels_name: path to testing labels
    :return: testing data
    """
    imgs = np.load(testing_data_name + ".npy")
    labels = np.load(testing_labels_name + ".npy")
    return imgs, labels
