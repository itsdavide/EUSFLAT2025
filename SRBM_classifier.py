#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the (alpha,beta)-SRBMs classifier appearing in the paper:
D. Petturiti and M. Rifqi (2025).
Alpha-Maxmin Classification with an Ensemble of Structural Restricted Boltzmann Machines.
"""

import numpy as np


# Class of a Structural Restricted Boltzmann Machine (SRBM)
class SRBM:
    # Constructor
    def __init__(self, num_classes, num_x, num_hidden, struct_perc):
        self.num_classes = num_classes
        self.num_x = num_x
        num_visible = num_classes + num_x
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.debug_print = True
        self.struct_perc = struct_perc
        
        # Fix the random seed
        np_rng = np.random.RandomState(1234)
        
        # Initalize the weight matrix
        self.weights = np.asarray(np_rng.uniform(
        			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))
        
        # Insert weights for the bias units into the first row and first column
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)
        
        # Create the random mask with the given struc_perc of zeros
        self.mask = np_rng.binomial(size=(num_visible, num_hidden), n = 1, p = 1 - self.struct_perc)
        
        # Insert weights for the bias units into the first row and first column
        self.mask = np.insert(self.mask, 0, 1, axis = 0)
        self.mask = np.insert(self.mask, 0, 1, axis = 1)
        
        
    # Logistic function   
    def sigma(self, x):
        return 1.0 / (1 + np.exp(-x))
        
    
    # Train the SRBM
    def train(self, data, max_epochs = 1000, learning_rate = 0.1):
        
        # Extract the number of examples
        num_examples = data.shape[0]
          
        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)
          
        # Execute Contrastive Divergenece (CD)
        for epoch in range(max_epochs):    
            # Positive CD phase
            pos_hidden_activations = np.dot(data, self.weights)      
            pos_hidden_probs = self.sigma(pos_hidden_activations)
            # Fix the bias unit
            pos_hidden_probs[:,0] = 1
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hidden_probs)
            
            # Negative CD phase
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self.sigma(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self.sigma(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            
            # Update weights
            self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)
            self.weights = self.weights * self.mask
            
            error = np.sum((data - neg_visible_probs) ** 2)
            if self.debug_print:
                print("Epoch %s: error is %s" % (epoch, error))
          
                
    # Extract the conditional probability distribution on the classes
    def prob_classifier(self, x_star):
        # Extarct the necessary weight matrices and bias vectors
        U = self.weights[1:self.num_classes+1, 1:self.num_hidden+2].T
        W = self.weights[self.num_classes+1:self.num_visible+2, 1:self.num_hidden+2].T
        c = self.weights[0:1, 1:self.num_hidden+2].squeeze()
        d = self.weights[1:self.num_classes+1, 0:1].squeeze()

        # Compute the conditional probabilites
        probs = np.zeros(self.num_classes)
        for y in range(self.num_classes):
            prod = np.exp(d[y])
            for j in range(self.num_hidden):
                tot = 0
                for k in range(self.num_x):
                    tot += W[j, k] * x_star[k] 
                prod *= 1 + np.exp(c[j] +  U[j, y] + tot)
            probs[y] = prod
        probs = probs / probs.sum()
              
        return probs
    