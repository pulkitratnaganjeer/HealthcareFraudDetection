import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ResponseEncoder(BaseEstimator, TransformerMixin):
    '''
    Class to do Response Encoding for the Categorical features.
    This class can be used in the sklearn's Pipeline to avoid data leakdage issues
    '''
    def __init__(self, categoricalFeatures, className):
        '''
        Function to initialize the class members
        
        Parameter(s):
        ------------
        categoricalFeatures: list
            List of features for which the response encoding has to be done to generate new features.
        className: str
            Name of the Class
        '''
        self.categoricalFeatures = categoricalFeatures # Categorical Features for which Response Encoding has to be done.
        self.responseTable = dict() # Dictionary to store the key:value pair with the 'key' being the categorical feature 
        # name and its 'value' as the dataFrame containing the Response Table.
        self.className = className
        self.classCount = 0 # Number of unique class labels. For binary classification, it will be 2.
        
    def fit(self, X, y):
        '''
        Function called on a Dataset (usually Train Dataset) and Class Label to generate Response Encoded Table.
        This function is called only for the train dataset and not for any cv/test dataset to avoid data leakage.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the Response Encoding has to be carried out.
        y: pandas.core.series.Series
            Class Labels of the DataFrame
        '''
        
        # Prepare a DataFrame based on the given the dataset and class label, containing just the input features and class labels
        data = pd.DataFrame()
        
        # Add the features to the dataframe
        for col in self.categoricalFeatures:
            data[col] = X[col]

        # Add the Class Label to the dataframe 
        data['PotentialFraud'] = y
        
        # Store the count of total unique class labels in the class variable 'classCount'
        self.classCount = len(y.unique())
        
        # Iterate through each of the categorical features for which Response Encoding has to be done
        for feature in self.categoricalFeatures:
            
            # Dictionary to store the unique categorical feature names and encoded feature name as keys and their values as
            # list of their corresponding values, class counts and count probabilities
            dictResponseTable = dict()
            
            uniqueFeatValues = np.sort(X[feature].unique()) # Array of unique feature values
            uniqueClassLabels = np.sort(y.unique()) # Array of unique class labels
            
            # Iterate through each of the categorical feature values and generate the Response Table
            for featureVal in uniqueFeatValues:
                
                countClass = list() # List to store the count/frequency of a Class label for a particular feature.
                probClass = list() # List to store the probability of occurence of a class label for a particular feature.
                
                # Loop through the unique Class Labels and find the count of the feature value
                for label in uniqueClassLabels:
                    
                    # Append the frequency of Class Label 'label' for the feature 'featureVal' to the list 'countClass'
                    countClass.append(data[(data[feature] == featureVal) & (data[self.className] == label)][self.className].count())
                    
                
                # Loop through the unique Class Labels and find the likelihood probability of the feature value
                for label in uniqueClassLabels:
                    
                    # Append the likelihood probability of the occurence of the class 'label' for the feature 'featureVal'
                    probClass.append(countClass[label]/sum(countClass))
                    
                
                # Prepare a dictionary having keys as features (original and new features) and their values as 
                # feature value (for original features), class counts and class probabilities
                
                # Check if the key already exist or not in the dictionary. If not, create it
                if (feature not in dictResponseTable.keys()):
                    dictResponseTable[feature] = []
                dictResponseTable[feature].append(featureVal) # Append the current iteration's feature value 'feature'
                
                # For each unique class label, add the class label and probability to the corresponding keys in the dictionary
                for label in uniqueClassLabels:
                    
                    if (feature + 'Class' + str(label) not in dictResponseTable.keys()):
                        dictResponseTable[feature + 'Class' + str(label)] = []
                    if (feature + '_' + str(label) not in dictResponseTable.keys()):
                        dictResponseTable[feature + '_' + str(label)] = []
                    dictResponseTable[feature + 'Class' + str(label)].append(countClass[label])
                    dictResponseTable[feature + '_' + str(label)].append(probClass[label])
                    
            # Prepare and store the Response Table in the dictionary 'self.responseTable'
            self.responseTable[feature] = pd.DataFrame(dictResponseTable)
        
        return self
    
    def transform(self, X, y= None):
        '''
        Function called on a Dataset (Train/Test Dataset) and/or Class Label to generate Response Encoded Features.
        This is called to avoid any data leakage. This uses the Response Table already prepared by the fit() method
        and does not consider the test dataset.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the Response Encoding has to be done.
        y: pandas.core.series.Series
            Class Labels of the DataFrame
        '''
        
        # Get a copy of the input dataframe such the input dataframe is not modified.
        xEncoded = X.copy()
        
        listResponseEncFeat = list() # List to store the names of the response encoded features.
        
        # Iterate through each of the categorical features for which Response Encoding has to be done
        for feature in self.categoricalFeatures:
        
            # Merge with the Response Table Dataframe
            xEncoded = pd.merge(left=xEncoded, right=self.responseTable[feature], how='left', on=feature)
            
            # Form the list of the features (original and class count) to be dropped from the Dataset
            listResponseEncFeat.extend([col for col in list(self.responseTable[feature].columns) if '_' not in col])
            
        # Fill the values for the datapoints which are not present in the Response Table with equal probabilities of the class
        xEncoded.fillna(1/self.classCount)
        
        # Drop the original features and class count features. Keep only the response encoded features having class prob.
        xEncoded.drop(columns=listResponseEncFeat, inplace=True)
        
        # Convert the values of the dataframe to numeric.
        xEncoded.apply(pd.to_numeric)
        
        # Fill the empty/missing values with 0.
        xEncoded.fillna(0, inplace=True)
        
        # Return this DataFrame with all the numerical features and the response encoded features for the categorical features
        return xEncoded