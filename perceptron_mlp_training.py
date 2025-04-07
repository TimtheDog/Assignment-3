#-------------------------------------------------------------------------
# AUTHOR: Ryan Dautel
# FILENAME: perceptron_mlp_training.py
# SPECIFICATION: Train and compare a Perceptron and an MLP on digit recognition
# FOR: CS 4210- Assignment #3
# TIME SPENT: 6 hours
#-------------------------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # learning rates
r = [True, False]  # shuffle values

# Reading the training data
df = pd.read_csv('optdigits.tra', sep=',', header=None)
X_training = np.array(df.values)[:, :64]
y_training = np.array(df.values)[:, -1]

# Reading the test data
df = pd.read_csv('optdigits.tes', sep=',', header=None)
X_test = np.array(df.values)[:, :64]
y_test = np.array(df.values)[:, -1]

# Initialize highest accuracy trackers
highest_perceptron_acc = 0
highest_mlp_acc = 0

# Iterate over all combinations of learning rate and shuffle
for learning_rate in n:
    for shuffle in r:
        for model_type in ['Perceptron', 'MLP']:
            if model_type == 'Perceptron':
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate,
                                    hidden_layer_sizes=(25,), shuffle=shuffle, max_iter=1000)

            # Train the model
            clf.fit(X_training, y_training)


            correct_predictions = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    correct_predictions += 1
            accuracy = correct_predictions / len(y_test)

            if model_type == 'Perceptron' and accuracy > highest_perceptron_acc:
                highest_perceptron_acc = accuracy
                print("Current Best Perceptron: "+str(accuracy)+", Parameters: learning rate="+str(learning_rate)+", shuffle="+str(shuffle))

            elif model_type == 'MLP' and accuracy > highest_mlp_acc:
                highest_mlp_acc = accuracy
                print("Current MLP Best accuracy: "+ str(accuracy) + ", Parameters: learning rate="+str(learning_rate)+", shuffle="+str(shuffle))
