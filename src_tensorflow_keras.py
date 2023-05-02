"""
Class for handing the NIH chest X-ray data. Designed to access images 
stored locally in a single directory.

Access the data:
Origin: https://nihcc.app.box.com/v/ChestXray-NIHCC
All images: https://rads-reader-images.s3.us-east-2.amazonaws.com/all_images/

"""

# Author: Justin Paul Turner <justin@justinpturner.com>
# License: BSD 3 clause
# Class for handing the chest X-ray data
# Designed to access data that is stored in a publicly available Amazon s3 bucket so that anyone can import and use the Rads_Reader class

    

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Rads_Reader:
    
    def __init__(self, 
                 condition = 'Cardiomegaly', 
                 bucket='rads-reader'):
        self.condition = condition
        self.bucket = bucket
    
    def ClassificationDf(self):
        df = pd.read_csv('https://rads-reader-images.s3.us-east-2.amazonaws.com/Image_Dateframe.csv', index_col = 0)
        return df

    def HalfConditionDf(self):
        df = self.ClassificationDf()
        nf_df = df[df['No Finding'] == 1].iloc[:sum(df[self.condition])] # Dataframe of no findings of the same length of the specified condition 
        condition_df = df[df[self.condition] == 1]
        half_condition_df = pd.concat([nf_df, condition_df])[['No Finding', self.condition]].sample(frac=1, random_state=0)
        return half_condition_df[['No Finding', self.condition]]
    
    def y(self): 
        return self.HalfConditionDf().to_numpy()
        
    def X(self): # The training array of half condition, half no_finding shuffled
        training_df = self.HalfConditionDf()
        X = pickle.load( open( "training_array.p", "rb" ) ).reshape((112120, 150, 150, 1))
        X_half_selection = X[[training_df.index]]
        return X_half_selection
    
    def SplitTrainingData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X(), self.y(), test_size=0.1, random_state=0)
        return X_train, X_test, y_train, y_test
    
    def BuildModel(self):
        model = Sequential()
        model.add(Conv2D(16, (3,3), activation='relu', input_shape = X_train[0].shape))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.3))

        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(2,2))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))


        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))
        
        METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'),
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc')
        ]
        model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=METRICS)
        return model
    
    
    
    

        
        
    
    
        
        
