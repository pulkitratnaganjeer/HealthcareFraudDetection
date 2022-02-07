import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class OneHotEncoder(BaseEstimator, TransformerMixin):
    '''
    Class to do One-hot Encoding for the Categorical features.
    This class can be used in the sklearn's Pipeline to avoid data leakdage issues
    '''
    def __init__(self, categoricalFeatures):
        '''
        Function to initialize the class members
        
        Parameter(s):
        ------------
        categoricalFeatures: list
            List of features for which the response encoding has to be done to generate new one-hot encoded features.
        '''
        self.categoricalFeatures = categoricalFeatures # Categorical Features for which One-hot Encoding has to be done.
        self.countVectorizers = dict() # Dictionary of CountVectorizer object for each features to be one-hot encoded.
        
        # Add CountVectorizer object for each feature to the dictionary 'self.countVectorizers'
        for feature in categoricalFeatures:
            
            # 'CountVectorizer not considering single letter text': https://stackoverflow.com/a/63339533/16007029
            self.countVectorizers[feature] = CountVectorizer(tokenizer=lambda x: x.split())
        
    def fit(self, X, y=None):        
        '''
        Function called on a Dataset (usually Train Dataset) to generate One-hot Encoded Features.
        This function is called only for the train dataset and not for any cv/test dataset to avoid data leakage.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the Response Encoding has to be carried out.
        '''
        
        # Fit the CountVectorizer object for each feature, on the Train Data feature
        for feature in self.categoricalFeatures:
            
            self.countVectorizers[feature].fit(pd.Series(X[feature]).astype(str).values)
            
        return self
    
    def transform(self, X, y=None):
        '''
        Function called on a Dataset (Train/Test Dataset) and/or Class Label to generate One-hot Encoded Features.
        This is called to avoid any data leakage. This uses the CountVectorizer object already prepared by the fit() method
        and does not consider the test dataset.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the One-hot Encoding has to be done.
        '''
        
        # Create a copy of the dataframe so that it does not modify the input dataframe
        xEncoded = X.copy()
        
        oneHotEncodedFeatures = dict() # Dictionary to store the feature name as key and its one-hot encoded csr matrix in its value
        
        # Transform each of the categorical feature to convert to one-hot encoding
        for feature in self.categoricalFeatures:
            
            oneHotEncodedFeatures[feature] = self.countVectorizers[feature].transform(pd.Series(xEncoded[feature]).astype(str).values)
            
        # Remove the original categorical features from the Dataset
        xEncoded.drop(columns=self.categoricalFeatures, inplace=True)
        
        # Stack (horizontally) the new one-hot encoded features to the dataset
        for feature in self.categoricalFeatures:
            
            xEncoded = hstack((xEncoded, oneHotEncodedFeatures[feature])).tocsr()
            
        # Return this DataFrame with all the numerical features and the one-hot encoded features for the categorical features
        return xEncoded
    '''
    Class to do One-hot Encoding for the Categorical features.
    This class can be used in the sklearn's Pipeline to avoid data leakdage issues
    '''
    def __init__(self, categoricalFeatures):
        '''
        Function to initialize the class members
        
        Parameter(s):
        ------------
        categoricalFeatures: list
            List of features for which the response encoding has to be done to generate new one-hot encoded features.
        '''
        self.categoricalFeatures = categoricalFeatures # Categorical Features for which One-hot Encoding has to be done.
        self.countVectorizers = dict() # Dictionary of CountVectorizer object for each features to be one-hot encoded.
        
        # Add CountVectorizer object for each feature to the dictionary 'self.countVectorizers'
        for feature in categoricalFeatures:
            
            # 'CountVectorizer not considering single letter text': https://stackoverflow.com/a/63339533/16007029
            self.countVectorizers[feature] = CountVectorizer(tokenizer=lambda x: x.split())
        
    def fit(self, X, y=None):        
        '''
        Function called on a Dataset (usually Train Dataset) to generate One-hot Encoded Features.
        This function is called only for the train dataset and not for any cv/test dataset to avoid data leakage.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the Response Encoding has to be carried out.
        '''
        
        # Fit the CountVectorizer object for each feature, on the Train Data feature
        for feature in self.categoricalFeatures:
            
            self.countVectorizers[feature].fit(pd.Series(X[feature]).astype(str).values)
            
        return self
    
    def transform(self, X, y=None):
        '''
        Function called on a Dataset (Train/Test Dataset) and/or Class Label to generate One-hot Encoded Features.
        This is called to avoid any data leakage. This uses the CountVectorizer object already prepared by the fit() method
        and does not consider the test dataset.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            DataFrame on which the One-hot Encoding has to be done.
        '''
        
        # Create a copy of the dataframe so that it does not modify the input dataframe
        xEncoded = X.copy()
        
        oneHotEncodedFeatures = dict() # Dictionary to store the feature name as key and its one-hot encoded csr matrix in its value
        
        # Transform each of the categorical feature to convert to one-hot encoding
        for feature in self.categoricalFeatures:
            
            oneHotEncodedFeatures[feature] = self.countVectorizers[feature].transform(pd.Series(xEncoded[feature]).astype(str).values)
            
        # Remove the original categorical features from the Dataset
        xEncoded.drop(columns=self.categoricalFeatures, inplace=True)
        
        # Stack (horizontally) the new one-hot encoded features to the dataset
        for feature in self.categoricalFeatures:
            
            xEncoded = hstack((xEncoded, oneHotEncodedFeatures[feature])).tocsr()
            
        # Return this DataFrame with all the numerical features and the one-hot encoded features for the categorical features
        return xEncoded