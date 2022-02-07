import numpy as np
import pandas as pd

def PreprocessData(xData):
    '''
    Function to implement the data pipeline for transforming the dataset into the required format as required by the
    Model.
    
    Parameters:
    ----------
    xData: DataFrame
        Dataset containing the features.
    '''
    
    # Create a copy of the dataset
    data = xData.copy()
    
    #region - Data Cleanup-------------------------------------------------------------------------------------------
    #================================================================================================================
    
    def dataCleanup(data):
        '''
        Cleans up the data in the given dataset and returns the cleaned dataset.

        Parameter:
        ---------
        data: pandas.core.frame.DataFrame
            Dataframe to be cleaned up
        '''

        #region - Beneficiary Columns-------------------------------------------------------------------------------

        # 'Renal Disease Indicator' column.
        # RenalDiseaseIndicator column has two unique values: 0 and 'Y'. Replace the value of 'Y' with 1.
        data['RenalDiseaseIndicator'].replace(to_replace='Y', value=1, inplace=True)
        
        # Convert the datatype of the 'RenalDiseaseIndicator' feature to numeric.
        data['RenalDiseaseIndicator'] = data['RenalDiseaseIndicator'].apply(pd.to_numeric)

        # 'Chronic Condition' columns
        # 'ChronicCond_' columns contains two unique values: 1 and 2. Replace the value of 2 with 0 to indicate 1 as 'Yes' 
        # and 0 as 'No'
        for chronicCol in [col for col in list(data.columns) if 'Chronic' in col]:
            data[chronicCol].replace(to_replace=2, value=0, inplace=True)

        # Rename the column 'County' as 'Country'
        data.rename(columns={'County': 'Country'}, inplace=True)    

        #endregion - Beneficiary Columns----------------------------------------------------------------------------


        #region - Inpatient and Outpatient Columns------------------------------------------------------------------

        # Fill empty values of the 'DeductibleAmtPaid' feature as it is a numeric feature and has some empty values.
        data['DeductibleAmtPaid'].fillna(value=0, inplace=True)

        #region - Inpatient and Outpatient Columns------------------------------------------------------------------


        #region - Other columns ------------------------------------------------------------------------------------

        # Drop the columns having all null values
        data.dropna(axis=1, how='all', inplace=True)

        # Replace the class label 'PotentialFraud' values. Replace 'Yes' with 1 and 'No' with 0.
        if ('PotentialFraud' in data.columns):

            data['PotentialFraud'].replace(to_replace=['Yes', 'No'], value=[1, 0], inplace=True)

        #endregion - Other columns ---------------------------------------------------------------------------------
        
    # Call the function 'dataCleanup' to clean the data in the dataframe
    dataCleanup(data)
        
    # Set of columns to be removed
    columnsToRemove = ['ClmProcedureCode_3', 'ClmProcedureCode_4', 'ClmProcedureCode_5']
    
    # Remove the above set of columns from the dataframe.
    data.drop(columns=columnsToRemove, inplace=True)
    
    #endregion - Data Cleanup----------------------------------------------------------------------------------------
    #================================================================================================================
    
    
    
    #region - Date Features------------------------------------------------------------------------------------------
    #================================================================================================================
    
    # Date related columns
    colDate = [col for col in data.columns if ('Dt' in col or 'DOB' in col or 'DOD' in col)]
    
    # Convert these columns to DateTime Format
    data[colDate] = data[colDate].apply(pd.to_datetime)
    
    # Generate new Feature 'ClaimSettlementDelay' (ClaimEndDt - ClaimStartDt)
    data['ClaimSettlementDelay'] = (data['ClaimEndDt'] - data['ClaimStartDt']).dt.days
    
    # Generate new Feature 'TreatmentDuration' (DischargeDt - AdmissionDt)
    data['TreatmentDuration'] = (data['DischargeDt'] - data['AdmissionDt']).dt.days
    data['TreatmentDuration'].fillna(0, inplace=True) # Filling empty values with 0 because the features 
    # 'DischargeDt' and 'AdmissionDt' exist only for Inpatient records.

    data['TreatmentDuration'] = data['TreatmentDuration'].apply(int)
    
    maxDate = max(data['ClaimEndDt'].max(), data['DischargeDt'].max())
    
    # Generate 'Age' feature from DOB based on the DOD or the maximum date.
    data['Age'] = data.apply(lambda x: round(((x['DOD'] - x['DOB']).days)/365) if pd.notnull(x['DOD'])
                             else round(((maxDate - x['DOB']).days)/365), axis=1)
    
    # Generate new Feature 'IsDead' based on whether there is a value in the DOD column or not
    data['IsDead'] = data['DOD'].apply(lambda x: 1 if pd.notnull(x) else 0)
    
    # Remove the set of date columns from the dataframe
    data.drop(columns=colDate, inplace=True)
    
    #endregion - Date Features---------------------------------------------------------------------------------------
    #================================================================================================================
    
    
    
    #region - Amount Features----------------------------------------------------------------------------------------
    #================================================================================================================
    
    # Fetch the features related to amount.
    colAmt = [col for col in data.columns if 'Amt' in col]
    
    # Total Claim Amount = Insurance Claim Amount reimbursed + Deductible Amount paid by the Subscriber
    data['TotalClaimAmount'] = data['InscClaimAmtReimbursed'] + data['DeductibleAmtPaid']

    # Total Inpatient Amount = Inpatient Annual Amount reimbursed + Inpatient Annual Deductible Amount
    data['IPTotalAmount'] = data['IPAnnualReimbursementAmt'] + data['IPAnnualDeductibleAmt']

    # Total Outpatient Amount = Outpatient Annual Amount reimbursed + Outpatient Annual Deductible Amount
    data['OPTotalAmount'] = data['OPAnnualReimbursementAmt'] + data['OPAnnualDeductibleAmt']
    
    # Remove the set of old amount features from the dataframe
    data.drop(columns=colAmt, inplace=True)
    
    #endregion - Amount Features-------------------------------------------------------------------------------------
    #================================================================================================================
    
    
    
    # Fetch the Claim Id and Provider Id from the dataset
    identifierData = data[['ClaimID', 'Provider']]
    
    # Drop some of the unique identifiers from the dataset as they won't contribute anything to do the classification.
    data.drop(columns=['ClaimID', 'BeneID', 'Provider', 'NoOfMonths_PartACov', 'NoOfMonths_PartBCov'], inplace=True)
    
    
    
    #region - Physician Features-------------------------------------------------------------------------------------
    #================================================================================================================
    
    # Fetch the columns related to Physicians
    colPhys = [col for col in data.columns if 'Physician' in col]
    
    # Prepare the feature 'UniquePhysCount'
    data['UniquePhysCount'] = data[colPhys].apply(lambda x: len(set([phys for phys in x if not pd.isnull(phys)])), axis=1)
    
    # Prepare the feature 'PhysRoleCount'
    data['PhysRoleCount'] = data[colPhys].apply(lambda x: len([phys for phys in x if not pd.isnull(phys)]), axis=1)
    
    # Prepare the feature 'IsSamePhysMultiRole1'
    data['IsSamePhysMultiRole1'] = data[['UniquePhysCount','PhysRoleCount']] \
                                .apply(lambda x: 1 if x['UniquePhysCount'] == 1 and x['PhysRoleCount'] > 1 else 0, axis=1)
    
    # Prepare the feature 'IsSamePhysMultiRole2'
    data['IsSamePhysMultiRole2'] = data[['UniquePhysCount','PhysRoleCount']] \
                                .apply(lambda x: 1 if x['UniquePhysCount'] == 2 and x['PhysRoleCount'] > 2 else 0, axis=1)
    
    def encodeCatFeatures(dataset, existingFeatures, newFeatures, suffix=''):
        '''
        Function to create new encoded features for Categorical Features, based on their count of values 
        in the existing set of features.

        Parameters:
        ----------
        dataset: pandas.core.frame.DataFrame
            DataFrame containing the data for which the new set of encoded features has to be created.
        exsitingFeatures: list
            List of existing features to considered for counting.
        newFeatures: list
            List of new features to encoded and created
        suffix: str
            Suffix to add before the new feature names.
        '''

        # Fetch the number of datapoints in the given dataset
        lenDatapoints = dataset.shape[0]

        # Iterate through each of the new features:
        for newFeature in newFeatures:

            listIsExistAllFeatures = list() # List to store a list of 0s and 1s for each existing feature,
            # if the new feature value exist in the existing features.

            # Iterate through each of the existing feature set and perform the logic to count.
            for existingFeature in existingFeatures:

                listIsExist = list() # List to store '1' if the new feature value exist in the existing feature.

                for value in list(dataset[existingFeature]):

                    if str(value) == str(newFeature):

                        listIsExist.append(1)

                    else:

                        listIsExist.append(0)

                listIsExistAllFeatures.append(listIsExist)

            arrayCount = np.zeros(lenDatapoints) # Array to store the count of the existing features containing the new features.

            # Iterate through each of the list in 'listIsExistAllFeatures' and sum the counts.
            for i in range(0, len(listIsExistAllFeatures)):

                arrayCount = arrayCount + np.array(listIsExistAllFeatures[i])

            dataset[suffix + newFeature] = arrayCount.astype(int)

        return dataset
    
    # Call the encodedFeatures function to generate new features: 'PHY412132', 'PHY337425', 'PHY330576'
    data = encodeCatFeatures(data, colPhys, ['PHY412132', 'PHY337425', 'PHY330576'])
    
    # Now remove the original features related to the Physicians
    data.drop(columns=['AttendingPhysician','OperatingPhysician','OtherPhysician'], inplace=True)
    
    #endregion - Physician Features----------------------------------------------------------------------------------
    #================================================================================================================
    
    
    
    #region - Claim Diagnosis Features-------------------------------------------------------------------------------
    #================================================================================================================
    
    # Fetch the columns related to the Claims Diagnosis Codes
    colDiagCode = [col for col in data.columns if 'ClmDiagnosisCode' in col]
    
    # Call the encodeCatFeatures function to generate new feature for the top 7 Claim Diagnosis Codes
    data = encodeCatFeatures(data, colDiagCode, ['4019', '2724', '42731', '25000', '2449', '53081', '4280'], 'ClmDiagCode_')
    
    # For each of the Claim Diagnosis Code Features, replace the values with 1 if there is a value, else replace with 0 .
    for diagCode in colDiagCode:
        data[diagCode] = data[diagCode].apply(lambda x: 1 if not pd.isnull(x) else 0)
    
    #endregion - Claim Diagnosis Features----------------------------------------------------------------------------
    #================================================================================================================
    
    
    
    #region - Claim Procedure Features-------------------------------------------------------------------------------
    #================================================================================================================
    
    # Fetch the columns related to the Claims Procedure Codes
    colProcCode = [col for col in data.columns if 'Procedure' in col]
    
    # Call the encodeCatFeatures function to generate new feature for the top 5 Claim Procedure Codes
    data = encodeCatFeatures(data, colProcCode, ['9904.0', '8154.0', '66.0', '3893.0', '3995.0'], 'ClmProcCode_')
    
    # For each of the Claim Procedure Code Features, replace the values with 1 if there is a value, else replace with 0 .
    for procCode in colProcCode:
        data[procCode] = data[procCode].apply(lambda x: 1 if not pd.isnull(x) else 0)
    
    #endregion - Claim Procedure Features----------------------------------------------------------------------------
    #================================================================================================================
    
    
    
    #region - Claim Admit Diagnosis Code and Diagnosis Group Code Features-------------------------------------------
    #================================================================================================================
    
    # For each of the Claim Admit Diagnosis Code and Diagnosis Group Code Features, 
    # replace the values with 1 if there is a value, else replace with 0 .
    for code in ['ClmAdmitDiagnosisCode', 'DiagnosisGroupCode']:
        data[code] = data[code].apply(lambda x: 1 if not pd.isnull(x) else 0)
    
    #endregion - Claim Admit Diagnosis Code and Diagnosis Group Code Features----------------------------------------
    #================================================================================================================
    
    
    
    #region - Gender Feature-----------------------------------------------------------------------------------------
    #================================================================================================================
    
    # Gender Feature has two values: 1 and 2. Replace 2 with 0.
    data['Gender'].replace(to_replace=2, value=0, inplace=True)
    
    #endregion - Gender Feature--------------------------------------------------------------------------------------
    #================================================================================================================
    
    # Remove some set of Claim Diagnosis/Procedure Code Features
    data.drop(columns=['DiagnosisGroupCode', 'ClmProcedureCode_1', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                       'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8'], 
              inplace=True)
    
    return data