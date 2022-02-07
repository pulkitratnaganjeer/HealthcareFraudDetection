import pandas as pd

def MergeDatasets(dataProvider, dataBeneficiary, dataInpatient, dataOutpatient):
        
    '''
    Combines the given datasets of Provider data, Beneficiary data, Inpatient and Outpatient claims data,
    and returns the combined dataset.

    Parameters:
    ----------
    dataProvider: pandas.core.frame.DataFrame
        DataFrame containing the Provider Unique Identifier.
    dataBeneficiary: pandas.core.frame.DataFrame
        DataFrame containing the Beneficiary related data.
    dataInpatient: pandas.core.frame.DataFrame
        DataFrame containing the Inpatient claims related data.
    dataOutpatient: pandas.core.frame.DataFrame
        DataFrame containing the Outpatient claims related data       
    '''

    # Concatenate the Inpatient and Outpatient dataset as these contain almost similar information
    dataConcat = pd.concat([dataInpatient, dataOutpatient])

    # Merge the above dataframe with the Beneficiary dataframe
    dataMerge = pd.merge(left=dataConcat, right=dataBeneficiary, on='BeneID')

    # Merge the above dataframe with the provider dataframe to add the Class Label.
    dataFinal = pd.merge(left=dataMerge, right=dataProvider, on='Provider')

    # Return the final dataframe
    return dataFinal