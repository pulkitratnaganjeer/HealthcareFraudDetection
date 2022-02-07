from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class Standardize(BaseEstimator, TransformerMixin):
    '''
    Class to do standardization of the numerical features.
    This class can be used in the sklearn's Pipeline to avoid data leakdage issues
    '''
    def __init__(self, numericalFeatures):
        '''
        Function to initialize the class members
        
        Parameter(s):
        ------------
        numericalFeatures: list
            List of numerical features to be standardized.
        '''
        self.numericalFeatures = numericalFeatures # Numerical Features to be standardized.
        self.standardScaler = StandardScaler() # Object of StandardScaler.
        
    def fit(self, X, y=None):        
        '''
        Function called on a Dataset (usually Train Dataset) to fit the train dataset.
        This function is called only for the train dataset and not for any cv/test dataset to avoid data leakage.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            Dataset to be considered for standardization.
        '''
        
        # Fit the StandardScaler object only on the numerical features.
        self.standardScaler.fit(X[self.numericalFeatures])        
            
        return self
    
    def transform(self, X, y=None):
        '''
        Function called on a Dataset (Train/Test Dataset) to standardize the data based on the Train Dataset.
        This is called to avoid any data leakage. This uses the standardScaler object already prepared by the fit() method
        and does not consider the test dataset.
        
        Parameters:
        ----------
        X: pandas.core.frame.DataFrame
            Dataset to be standardized.
        '''
        
        # Create a copy of the dataframe so that it does not modify the input dataframe
        xStandardized = X.copy()
        
        # Standardize the numerical features of the dataset.
        xStandardized[self.numericalFeatures] = self.standardScaler.transform(xStandardized[self.numericalFeatures])
            
        # Return the standardized dataset
        return xStandardized