#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the (alpha,beta)-SRBMs classifier appearing in the paper:
D. Petturiti and M. Rifqi (2025).
Alpha-Maxmin Classification with an Ensemble of Structural Restricted Boltzmann Machines.
"""

import numpy as np
import pandas as pd
from SRBM_classifier import SRBM
from sklearn.model_selection import StratifiedKFold

# Stratified 5-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fix the number of binary variables in the dataset and the number of classes
num_x = 20
num_classes = 5
num_visible = num_x + num_classes

# Load the dataset
folder = 'datasets/p_0.25/'
df = pd.read_csv(folder + 'dataset_' + str(num_x) + '_bin_x_' + str(num_classes) + '_c.csv')
x_labels = ['x' + str(i) for i in range(1, num_x + 1, 1)]
y_labels = ['y' + str(i) for i in range(1, num_classes + 1, 1)]
X_y_oh_df = df[x_labels + y_labels]
y = df[['y']].squeeze()
       

# Execute the stratified 5-fold cross validation
accs = []
for i, (train_index, test_index) in enumerate(skf.split(X_y_oh_df, y)):
    
    # Extract training and test sets
    # (The dataset is assumed to be in the form of num_x binary variables
    # and num_classes binary columns for the one-hot encoding of the class)
    print(f'Fold {i}:')
    X_train = np.array(X_y_oh_df[y_labels + x_labels].iloc[train_index])
    X_test = np.array(X_y_oh_df[y_labels + x_labels].iloc[test_index])
    print('Train set size:', len(X_train))
    print('Test set size:', len(X_test))
    
    
    # Create the ensemble of SRBMs
    srbms = []
    #
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = num_visible, struct_perc = 0.05))
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = num_visible, struct_perc = 0.1))
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = num_visible, struct_perc = 0.15))
    #
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = int(num_visible * (1 + 1/3)), struct_perc = 0.05))
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = int(num_visible * (1 + 1/3)), struct_perc = 0.1))
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = int(num_visible * (1 + 1/3)), struct_perc = 0.15))
    #
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = int(num_visible * (1 + 2/3)), struct_perc = 0.05))
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = int(num_visible * (1 + 2/3)), struct_perc = 0.1))
    srbms.append(SRBM(num_x = num_x, num_classes = num_classes, num_hidden = int(num_visible * (1 + 2/3)), struct_perc = 0.15))

    # Train the ensemble
    for r in srbms:
        r.debug_print = False
        r.train(X_train, max_epochs = 1000, learning_rate = 0.15)

    # Test the ensemble
    tot = 0
    for row in X_test:
        # Decompose the test set 
        y_star = np.array(row[0:num_classes])
        x_star = np.array(row[num_classes:num_x + num_classes + 1])
        
        # Extract the conditional distributions on classes from each SRBM
        distribs = []

        for r in srbms:
            distrib = r.prob_classifier(x_star)
            distribs.append(distrib)
    
        distribs = np.array(distribs)
      
        # Compute the reference distribution according to L2 distance (centroid)
        centroid = distribs.sum(axis=0) / distribs.shape[0]

        # Compute the L2 distances from the centroid
        distances = np.zeros(distribs.shape[0])
        for i in range(distribs.shape[0]):
            distances[i] = ((distribs[i] - centroid)**2).sum() / distribs.shape[1]
                
        # Compute the beta-quantile of distances
        beta = 1 - (2 /  distribs.shape[0])
        quantile = np.quantile(distances, beta)
              
        # Filters the distributions in the credal classifier from outliers
        mask_dist = distances <= quantile
               
        # Compute the lower and upper envelopes of the filtered credal classifier
        min_dist = distribs[mask_dist].min(axis=0)
        max_dist = distribs[mask_dist].max(axis=0)
             
        # Compute the alpha-JP mixture of the envelopes
        alpha = 0.5
        alpha_mix = alpha * min_dist + (1 - alpha) * max_dist

        # Find the optimal class according to the alpha-JP mixture
        i_star = np.argmax(alpha_mix).squeeze()

        # Keep track for accuracy
        # (classes are renumbered from 0 to c-1 to coincide with indices)  
        tot += y_star[i_star]
    
    # Compute the accuracy for this fold
    print('Accuracy:', np.round(tot / len(X_test) * 100.0, 1), '%')
    accs.append(tot / len(X_test))
    print()
    
# Compute the mean and the variance of accuracy in the 5 folds
accs = np.array(accs)
print('*** Result in the 5 folds ***')
print('Mean accuracy:', np.round(accs.mean() * 100.0, 1), '%')
print('Std accuracy:', np.round(accs.std() * 100.0, 1), '%')