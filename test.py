import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
import pickle

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
    
    
# Read in cars and notcars
car_images = glob.glob('labelled_imgs/vehicles/*.png')
notcar_images = glob.glob('labelled_imgs/non-vehicles/*.png')
cars = []
notcars = []
for image in car_images:
    cars.append(image)
    
for image in notcar_images:
    notcars.append(image)

sample_size = 500
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

with open("model.p", "rb") as ifh:
    parameters = pickle.load(ifh)
    X_scaler = pickle.load(ifh)
    svc = pickle.load(ifh)
    
    
    car_features = extract_features(cars, **parameters)
    notcar_features = extract_features(notcars, **parameters)
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    X_test = X_scaler.transform(X_test)
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255



