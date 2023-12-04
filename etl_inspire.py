import pandas as pd
import numpy as np
from datetime import datetime
import os, time


def concept_mapping(input_path, mapped_path):
    '''
    Map source concepts from source tables to standard concepts.
    Returns mapped results of source tables.
    input_path : input path for source tables (.csv)
    Download and place 'parameters_inspire_mapped.csv' into 'input_path/mapped/' directory.
    '''
    
    print('@@@ Concept mapping for INSPIRE dataset @@@')
    # INSPIRE v0.2 (about 130,000 cases, 50% of the surgical cases)
    # Load the source tables within INSPIRE v2 into dataframes
    print('Loading source tables...', end='')
    df_diag = pd.read_csv(f'{input_path}/diagnosis.csv')         # Load diagnosis data
    df_labs = pd.read_csv(f'{input_path}/labs.csv')              # Load labs data
    df_medi = pd.read_csv(f'{input_path}/medications.csv')       # Load medications data
    df_op = pd.read_csv(f'{input_path}/operations.csv')          # Load operations data
    df_vitals = pd.read_csv(f'{input_path}/vitals.csv')          # Load vitals data
    df_ward = pd.read_csv(f'{input_path}/ward_vitals.csv')       # Load ward vitals data
    
    # Load the CONCEPT_RELATIONSHIP table with tab as a delimiter and error handling for bad lines
    df_concept_rel = pd.read_csv(f'vocab/CONCEPT_RELATIONSHIP.csv', sep='\t', on_bad_lines='error')
    # LOAD the CONCEPT table with tab as a delimiter
    df_concept = pd.read_csv(f'vocab/CONCEPT.csv', sep='\t')
    print('done')    
        
    # Display the number of records in each dataset
    print(f'Size of the tables: operations {len(df_op)}, diagnosis {len(df_diag)}, labs {len(df_labs)}, medications {len(df_medi)}, vitals {len(df_vitals)}, ward_vitals {len(df_ward)}')

    # Display the total unique subjects present in the combined dataset
    print(f"total subjects in operations.csv: {len(np.unique(df_op['subject_id']))}")


    ### diagnosis table ###
    # Create a filtered DataFrame for 'Maps to' relationships
    maps_to_df = df_concept_rel[df_concept_rel['relationship_id'] == 'Maps to']

    # Merge df_data with df_concept to get source_concept_id
    merged_data = df_diag.merge(df_concept[df_concept['vocabulary_id'] == 'ICD10'], left_on='icd10_cm', right_on='concept_code', how='left')

    # Merge the result with maps_to_df to get standard_concept_id
    df_diag_mapped = merged_data.merge(maps_to_df, left_on='concept_id', right_on='concept_id_1', how='left')

    # Populate the 'source_concept_id', and 'standard_concept_id' columns in the df_op dataframe
    df_diag['source_concept_id'] = df_diag_mapped['concept_id']
    df_diag['standard_concept_id'] = df_diag_mapped['concept_id_2']

    # Calculate and print the number of DIAGNOSIS table records that couldn't be mapped to a standard concept
    nan_sum = df_diag['standard_concept_id'].isna().sum()
    print(f'Mismatched concepts in DIAGNOSIS table: {len(df_diag)} ({nan_sum/len(df_diag)*100:.1f}%)')

    df_diag.rename(columns = {'icd10_cm' : 'source_value'}, inplace=True)

    # Remove rows from the df_op dataframe where the 'standard_concept_id' is missing
    df_diag.dropna(subset= 'standard_concept_id', inplace=True, ignore_index=True)


    ### medication table ###
    # Splitting combined drug names and creating separate rows for each drug
    df_medi = df_medi.assign(drug_name=df_medi['drug_name'].str.split('/')).explode('drug_name').reset_index(drop=True)

    # Filtering the standard concepts related to medications from RxNorm and RxNorm Extension vocabularies
    df_medi_concept = df_concept[((df_concept['vocabulary_id'] == 'RxNorm') | (df_concept['vocabulary_id'] == 'RxNorm Extension')) & (df_concept['standard_concept'] == 'S')][['concept_name', 'concept_id']]

    # Converting the concept names to lowercase for easier matching
    df_medi_concept['concept_name'] = df_medi_concept['concept_name'].str.lower()

    # Merging the medications data with the standard drug concepts based on drug names
    df_medi = df_medi.merge(df_medi_concept, left_on='drug_name', right_on='concept_name', how='left')

    # Calculate and print the number of MEDICATIONS table records that couldn't be mapped to a standard concept
    nan_sum = df_medi['concept_id'].isna().sum()
    print(f'Mismatched concepts in MEDICATIONS table: {nan_sum} / {len(df_medi)} ({nan_sum/len(df_medi)*100:.1f}%)')


    ### labs table ###
    # Load PARAMETERS table that contains manually mapped results of item_names used in source tables
    df_params = pd.read_csv(f'{mapped_path}/parameters_inspire_mapped.csv')

    # Merge 'df_labs' with 'df_params' using 'item_name' and 'Label' as keys to map concepts
    df_labs = df_labs.merge(df_params[['Label', 'Unit', 'concept_id']], left_on = 'item_name', right_on = 'Label', how='left')
    # Drop the redundant 'Label' column post merging
    df_labs.drop(columns='Label', inplace=True)

    # Calculate and print the number of LABS table records that couldn't be mapped to a standard concept
    nan_sum = df_labs['concept_id'].isna().sum()
    print(f'Mismatched concepts in LABS table: {nan_sum} / {len(df_labs)} ({nan_sum/len(df_labs)*100:.1f}%)')


    ### vitals table ###
    # Merge 'df_vitals' with 'df_params' using 'item_name' and 'Label' as keys to map concepts
    df_vitals = df_vitals.merge(df_params.loc[df_params['Table']=='vitals', ['Label', 'Unit', 'concept_id', 'vocab']], left_on = 'item_name', right_on = 'Label', how='left')
    # Drop the redundant 'Label' column post merging
    df_vitals.drop(columns='Label', inplace=True)

    # Filter out rows where 'item_name' is either 'cpat' or 'ds'
    df_vitals = df_vitals[~df_vitals['item_name'].isin(['cpat', 'ds'])]

    # Calculate and print the number of VITALS table records that couldn't be mapped to a standard concept
    nan_sum = df_vitals['concept_id'].isna().sum()
    print(f'Mismatched concepts in VITALS table: {nan_sum} / {len(df_vitals)} ({nan_sum/len(df_vitals)*100:.1f}%)')


    ### ward_vitals table ###
    # Merge 'df_ward' with 'df_params' using 'item_name' and 'Label' as keys to map concepts
    df_ward = df_ward.merge(df_params.loc[df_params['Table']=='ward_vitals', ['Label', 'concept_id']], left_on = 'item_name', right_on = 'Label')
    # Drop the redundant 'Label' column post merging
    df_ward.drop(columns='Label', inplace=True)

    # Calculate and print the number of WARD_VITALS table records that couldn't be mapped to a standard concept
    nan_sum = df_ward['concept_id'].isna().sum()
    print(f'Mismatched concepts in WARD_VITALS table: {nan_sum} / {len(df_ward)} ({nan_sum/len(df_ward)*100:.1f}%)')


    ### operation table ###
    # Filter the CONCEPT_RELATIONSHIP table to only include 'Maps to' relationships
    maps_to_df = df_concept_rel[df_concept_rel['relationship_id'] == 'Maps to']

    # Merge the OPERATION table with the CONCEPT table using the ICD10PCS vocabulary to obtain the source_concept_id
    merged_data = df_op.merge(df_concept[df_concept['vocabulary_id'] == 'ICD10PCS'], left_on='icd10_pcs', right_on='concept_code', how = 'left')

    # Merge the resulting data with the 'maps_to' relationship data to obtain the corresponding standard_concept_id
    df_op_mapped = merged_data.merge(maps_to_df, left_on='concept_id', right_on='concept_id_1', how='left')

    # Populate the 'source_value', 'source_concept_id', and 'standard_concept_id' columns in the df_op dataframe
    df_op['source_value'] = df_op_mapped['icd10_pcs']
    df_op['source_concept_id'] = df_op_mapped['concept_id']
    df_op['standard_concept_id'] = df_op_mapped['concept_id_2']

    # Calculate and print the number of OPERATION table records that could not be mapped to a standard concept
    nan_sum = df_op['standard_concept_id'].isna().sum()
    print(f'Mismatched concepts in OPERATION table: {nan_sum} / {len(df_op)} ({nan_sum/len(df_op)*100:.1f}%)')

    # Remove rows from the df_op dataframe where the 'standard_concept_id' is missing
    df_op.dropna(subset= 'standard_concept_id', inplace=True, ignore_index=True)
    
    return df_diag, df_labs, df_medi, df_op, df_vitals, df_ward, df_params


def table_mapping(source_df, save_path, save_csv=False):
    '''
    Map source tables to OMOP-CDM standardized tables. 
    Saves the mapped cdm tables in parquet.
    source_df : a tuple of source tables in dataframe
    save_path : path to save the mapped cdm tables
    save_csv : If True, also save the cdm tables into csv.
    '''
    
    # Make directory to save the results
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.makedirs(os.path.join(save_path, 'sample'), exist_ok=True)
    
    # start_index for each table_id
    start_index = {
    'person': 2000000,
    'observation_period': 2000000,
    'visit_occurrence': 2000000,
    'visit_detail': 2000000,
    'condition_occurrence': 2000000,
    'drug_exposure': 2000000,
    'procedure_occurrence': 2000000,
    'measurement': 2000000,
    'note': 2000000,
    'location': 2000000 
    }
    # name of source dataset
    dname = 'INSPIRE'
    
    # Load source tables
    df_diag, df_labs, df_medi, df_op, df_vitals, df_ward, df_params = source_df
    print('@@@ Table mapping for INSPIRE dataset @@@')

    ### PERSON TABLE ###
    print('PERSON TABLE...', end='')
    # Create an empty dataframe for PERSON table
    df_person = pd.DataFrame(columns=['PERSON_ID'])

    # Assign unique IDs to each distinct 'subject_id' from the operations data
    unique_ids = df_op['subject_id'].unique()
    df_person['PERSON_ID'] = start_index['person'] + np.arange(len(unique_ids)) + 1
    df_person['subject_id'] = unique_ids

    # Merge relevant columns from the operations dataframe with the PERSON dataframe based on 'subject_id'
    usecols = ['subject_id', 'age', 'sex', 'race']
    df_person = df_person.merge(df_op[usecols], on = 'subject_id')
    # Ensure only the latest discharge_time is retained for each unique PERSON_ID
    df_person.drop_duplicates(subset = 'PERSON_ID', keep = 'first', inplace = True, ignore_index = True)

    # Map gender values ('M' or 'F') to corresponding GENDER_CONCEPT_ID values
    df_person['GENDER_CONCEPT_ID'] = df_person['sex'].map({'M': 8507, 'F': 8532}, na_action='ignore')
    # Remove any rows with missing gender values
    df_person.dropna(subset=['GENDER_CONCEPT_ID'])

    # Set the first date of all patients to 2011.01.01 since the exact year is not specified
    start_date = datetime(2011, 1, 1)

    # Calculate and assign the year of birth based on age and the start date
    df_person['YEAR_OF_BIRTH'] = start_date.year - df_person['age']
    # Compute the exact birth datetime using age and start date
    df_person['BIRTH_DATETIME'] = pd.to_datetime(start_date) - pd.to_timedelta(df_person['age']*365.25, unit = 'days')

    # Set RACE_CONCEPT_ID to indicate all individuals are ASIAN
    #df_person['RACE_CONCEPT_ID'] = 8515
    
    # Assign value for LOCATION_ID (2: INSPIRE)
    df_person['LOCATION_ID'] = 'inspire'

    # Populate source value columns based on values from the operations data
    df_person['PERSON_SOURCE_VALUE'] = df_person['subject_id']
    df_person['GENDER_SOURCE_VALUE'] = df_person['sex']
    #df_person['RACE_SOURCE_VALUE'] = df_person['race']
    #df_person['RACE_SOURCE_CONCEPT_ID'] = 8515

    # Remove columns that aren't part of the final PERSON table format
    df_person.drop(columns=usecols, inplace=True)

    # Write the processed data to a parquet file
    df_person.to_parquet(f'{save_path}/{dname}_PERSON.parquet')
    if save_csv:
        df_person.to_csv(f'{save_path}/{dname}_PERSON.csv', index=False)
    df_person[:1000].to_csv(f'{save_path}/sample/{dname}_PERSON.csv', index=False)    
    print('done')


    ### OBSERVATION_PERIOD ###
    print('OBSERVATION_PERIOD TABLE...', end='')
    # Create an empty dataframe for OBSERVATION_PERIOD table
    df_obs = pd.DataFrame(columns=['OBSERVATION_PERIOD_ID'])

    # Copy PERSON_ID from PERSON table to OBSERVATION_PERIOD table
    df_obs['OBSERVATION_PERIOD_ID'] =  start_index['observation_period'] - start_index['person'] + df_person['PERSON_ID']
    # Assign OBSERVATION_PERIOD_ID to each PERSON_ID
    df_obs['PERSON_ID'] = df_person['PERSON_ID']
    # Copy PERSON_SOURCE_VALUE from PERSON table to subject_id in OBSERVATION_PERIOD table for merging purposes
    df_obs['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Define the base date for the observation period
    base_date = datetime(2011, 1, 1)

    # Merge the 'discharge_time' column from the operations dataframe to the OBSERVATION_PERIOD table
    df_obs = df_obs.merge(df_op[['subject_id', 'discharge_time']], on='subject_id')
    # Retain only the latest 'discharge_time' for each subject
    df_obs.drop_duplicates(subset='subject_id', keep='last', inplace=True, ignore_index=True)

    # Aggregate the maximum chart time across all source tables (diagnosis, labs, medications, vitals, ward)
    # This helps in determining the end of the observation period for each subject
    df_time = df_obs[['subject_id', 'discharge_time']].merge(df_diag[['subject_id', 'chart_time']], on='subject_id', how = 'left').drop_duplicates(subset='subject_id', keep = 'last', inplace = False, ignore_index = True)
    df_time = df_time.merge(df_labs[['subject_id', 'chart_time']], on = 'subject_id', how = 'left', suffixes=("_diag", "_labs")).drop_duplicates(subset='subject_id', keep = 'last', inplace = False, ignore_index = True)
    df_time = df_time.merge(df_medi[['subject_id', 'chart_time']], on = 'subject_id', how = 'left', suffixes=(None, "_medi")).drop_duplicates(subset='subject_id', keep = 'last', inplace = False, ignore_index = True)
    df_time = df_time.merge(df_vitals[['subject_id', 'chart_time']], on = 'subject_id', how = 'left', suffixes=(None, "_vitals")).drop_duplicates(subset='subject_id', keep = 'last', inplace = False, ignore_index = True)
    df_time = df_time.merge(df_ward[['subject_id', 'chart_time']], on = 'subject_id', how = 'left', suffixes=(None, "_ward")).drop_duplicates(subset='subject_id', keep = 'last', inplace = False, ignore_index = True)
    df_time['max_time'] = df_time.iloc[:,1:].max(axis=1)

    # Set the OBSERVATION_PERIOD_START_DATE to the base date
    df_obs['OBSERVATION_PERIOD_START_DATE'] = pd.to_datetime(base_date)
    # Calculate and set the OBSERVATION_PERIOD_END_DATE using the base date and the aggregated maximum time
    df_obs['OBSERVATION_PERIOD_END_DATE'] = pd.to_datetime(base_date) + pd.to_timedelta(df_time['max_time'], unit='min')
    # Convert the OBSERVATION_PERIOD_END_DATE to just date format (remove time)
    df_obs['OBSERVATION_PERIOD_END_DATE'] = df_obs['OBSERVATION_PERIOD_END_DATE'].dt.date

    # Assign the PERIOD_TYPE_CONCEPT_ID indicating the data source is an EHR since it is not specified
    df_obs['PERIOD_TYPE_CONCEPT_ID'] = 32817

    # Remove columns that aren't part of the final OBSERVATION_PERIOD table format
    df_obs.drop(columns=['discharge_time', 'subject_id'], inplace=True)

    # Write the processed data to a parquet file
    df_obs.to_parquet(f'{save_path}/{dname}_OBSERVATION_PERIOD.parquet')
    if save_csv:
        df_obs.to_csv(f'{save_path}/{dname}_OBSERVATION_PERIOD.csv', index=False)
    df_obs[:1000].to_csv(f'{save_path}/sample/{dname}_OBSERVATION_PERIOD.csv', index=False)    
    print('done')


    ### VISIT_OCCURRENCTE TABLE ###
    # Match admission records
    print('VISIT_OCCURRENCE TABLE...', end='')
    # Create an empty dataframe for VISIT_OCCURRENCE table
    df_visit_occ = pd.DataFrame(columns=['VISIT_OCCURRENCE_ID'])

    # Copy PERSON_ID values from df_person to df_visit_occ
    df_visit_occ['PERSON_ID'] = df_person['PERSON_ID']
    # Copy PERSON_SOURCE_VALUE values (as subject_id) from df_person to df_visit_occ
    df_visit_occ['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Merge visit_occurrence data with operation data based on 'subject_id'
    usecols = ['hadm_id', 'subject_id', 'admission_time', 'discharge_time']
    df_visit_occ = df_visit_occ.merge(df_op[usecols], on='subject_id')
    # Remove duplicate entries based on 'hadm_id', keeping only the first occurrence
    df_visit_occ.drop_duplicates(subset=['hadm_id'], keep='first', inplace=True, ignore_index=True)
    # Assign sequential IDs starting from 1 to VISIT_OCCURRENCE_ID column
    df_visit_occ['VISIT_OCCURRENCE_ID'] = start_index['visit_occurrence'] + np.arange(len(df_visit_occ)) + 1

    # Set a default value for VISIT_CONCEPT_ID
    df_visit_occ['VISIT_CONCEPT_ID'] = 9201

    # Define the base date
    base_date = datetime(2011, 1, 1)
    # Calculate and assign VISIT_START_DATETIME based on admission time in minutes from the base date
    df_visit_occ['VISIT_START_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_visit_occ['admission_time'], unit='min')
    # Extract the date part for VISIT_START_DATE
    df_visit_occ['VISIT_START_DATE'] = pd.to_datetime(df_visit_occ['VISIT_START_DATETIME'].dt.date)
    # Calculate and assign VISIT_END_DATETIME based on discharge time in minutes from the base date
    df_visit_occ['VISIT_END_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_visit_occ['discharge_time'], unit='min')
    # Extract the date part for VISIT_END_DATE
    df_visit_occ['VISIT_END_DATE'] = pd.to_datetime(df_visit_occ['VISIT_END_DATETIME'].dt.date)

    # Assign the VISIT_TYPE_CONCEPT_ID indicating the data source is an EHR since it is not specified
    df_visit_occ['VISIT_TYPE_CONCEPT_ID'] = 32817

    ## Mapping PRECEDING_VISIT_OCCURRENCE_ID
    # Generate a column with the previous 'subject_id' for determining preceding visit occurrence
    df_visit_occ['prev_subject_id'] = df_visit_occ['subject_id'].shift(1).astype('Int64')
    # Create a new boolean column 'nadm' to check if the current row's subject_id matches the previous one
    df_visit_occ['nadm'] = df_visit_occ['subject_id'] == df_visit_occ['prev_subject_id']
    # Set the first row's 'nadm' value to False since there's no preceding record
    df_visit_occ.at[0, 'nadm'] = False 
    # Compute PRECEDING_VISIT_OCCURRENCE_ID based on 'nadm'
    df_visit_occ['PRECEDING_VISIT_OCCURRENCE_ID'] = np.where(df_visit_occ['nadm'], df_visit_occ['VISIT_OCCURRENCE_ID'].shift(1), np.nan)
    df_visit_occ['PRECEDING_VISIT_OCCURRENCE_ID'] = df_visit_occ['PRECEDING_VISIT_OCCURRENCE_ID'].astype('Int64')
    
     # Remove columns that aren't part of the final VISIT_OCCURRENCE table format except hadm_id
    df_visit_occ.drop(columns=usecols[1:], inplace=True)


    ### VISIT_DETAIL TABLE ###
    # Match ICU_ADMIN record
    print('VISIT_DETAIL TABLE...', end='')
    # Create a new DataFrame for VISIT_DETAIL data with the specified columns
    df_visit_detail = pd.DataFrame(columns=['VISIT_DETAIL_ID'])

    # Populate the PERSON_ID and subject_id columns with data from the df_person DataFrame
    df_visit_detail['PERSON_ID'] = df_person['PERSON_ID']
    df_visit_detail['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Integrate visit detail data with operational data from df_op using 'subject_id'
    usecols = ['hadm_id', 'subject_id', 'icuin_time', 'icuout_time']
    df_visit_detail = df_visit_detail.merge(df_op[usecols], on='subject_id')

    # Remove duplicate visit records based on hospital admission ID and ICU admission time
    df_visit_detail.drop_duplicates(subset=['hadm_id', 'icuin_time'], keep='first', inplace=True, ignore_index=True)
    # Exclude rows with missing ICU admission time
    df_visit_detail.dropna(subset='icuin_time', inplace=True, ignore_index=True)

    # Generate unique sequential IDs for VISIT_DETAIL_ID
    df_visit_detail['VISIT_DETAIL_ID'] = start_index['visit_detail'] + np.arange(len(df_visit_detail)) + 1

    # Designate a concept ID representing ICU visits
    df_visit_detail['VISIT_DETAIL_CONCEPT_ID'] = 32037

    # Calculate visit start and end datetime values using base_date and ICU admission/discharge times
    base_date = datetime(2011, 1, 1)
    df_visit_detail['VISIT_DETAIL_START_DATETIME'] = base_date + pd.to_timedelta(df_visit_detail['icuin_time'], unit='min')
    df_visit_detail['VISIT_DETAIL_START_DATE'] = pd.to_datetime(df_visit_detail['VISIT_DETAIL_START_DATETIME'].dt.date)
    df_visit_detail['VISIT_DETAIL_END_DATETIME'] = base_date + pd.to_timedelta(df_visit_detail['icuout_time'], unit='min')
    df_visit_detail['VISIT_DETAIL_END_DATE'] = pd.to_datetime(df_visit_detail['VISIT_DETAIL_END_DATETIME'].dt.date)

    # Specify the concept ID for the visit detail type as sourced from EHR
    df_visit_detail['VISIT_DETAIL_TYPE_CONCEPT_ID'] = 32817

    # Determine preceding visits for each entry by comparing 'hadm_id' with its previous entry
    df_visit_detail['prev_hadm_id'] = df_visit_detail['hadm_id'].shift(1).astype('Int64')
    df_visit_detail['nadm'] = df_visit_detail['hadm_id'] == df_visit_detail['prev_hadm_id']
    df_visit_detail.at[0, 'nadm'] = False  # The first entry won't have a preceding visit
    df_visit_detail['PRECEDING_VISIT_DETAIL_ID'] = np.where(df_visit_detail['nadm'], df_visit_detail['VISIT_DETAIL_ID'].shift(1), np.nan)
    df_visit_detail['PRECEDING_VISIT_DETAIL_ID'] = df_visit_detail['PRECEDING_VISIT_DETAIL_ID'].astype('Int64')

    # Merge with df_visit_occ to fetch 'VISIT_OCCURRENCE_ID' values
    df_visit_detail['VISIT_OCCURRENCE_ID'] = df_visit_detail.merge(df_visit_occ[['hadm_id', 'VISIT_OCCURRENCE_ID']], on='hadm_id', how='left')['VISIT_OCCURRENCE_ID']

    # Remove columns that aren't part of the final VISIT_DETAIL table
    df_visit_detail.drop(columns=usecols[1:], inplace=True)


    ### CONDITION_OCCURRENCE TABLE ###
    print('CONDITION_OCCURRENCE TABLE...', end='')
    # Create an empty DataFrame using the predefined column names
    df_cond_occ = pd.DataFrame(columns=['CONDITION_OCCURRENCE_ID'])

    # Map PERSON_ID values from the df_person DataFrame to the new CONDITION_OCCURRENCE DataFrame
    df_cond_occ['PERSON_ID'] = df_person['PERSON_ID']

    # Transfer subject_id values (stored as PERSON_SOURCE_VALUE) from df_person to df_cond_occ
    df_cond_occ['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Merge the df_cond_occ DataFrame with df_diag using the 'subject_id' as a common column
    df_cond_occ = df_cond_occ.merge(df_diag, on='subject_id', how = 'left')

    # Generate unique IDs for each row in the CONDITION_OCCURRENCE table
    df_cond_occ['CONDITION_OCCURRENCE_ID'] = start_index['condition_occurrence'] + np.arange(len(df_cond_occ)) + 1

    # Rename columns to match the target column names in the CONDITION_OCCURRENCE table
    df_cond_occ.rename(columns={'standard_concept_id': 'CONDITION_CONCEPT_ID', 
                                'source_value': 'CONDITION_SOURCE_VALUE', 
                                'source_concept_id': 'CONDITION_SOURCE_CONCEPT_ID'}, inplace=True)

    # Establish a reference starting date for generating dates in the observation period
    base_date = datetime(2011, 1, 1)

    # Convert 'chart_time' values (in minutes) to datetime objects, with the reference as the base_date
    df_cond_occ['CONDITION_START_DATETIME'] = base_date + pd.to_timedelta(df_cond_occ['chart_time'], unit='min')
    df_cond_occ['CONDITION_START_DATE'] = pd.to_datetime(df_cond_occ['CONDITION_START_DATETIME'].dt.date)

    # Set end dates equal to start dates as there's no separate end time
    df_cond_occ['CONDITION_END_DATETIME'] = df_cond_occ['CONDITION_START_DATETIME']
    df_cond_occ['CONDITION_END_DATE'] = df_cond_occ['CONDITION_START_DATE']

    # Assign the CONDITION_TYPE_CONCEPT_ID indicating the data source is an EHR since it is not specified
    df_cond_occ['CONDITION_TYPE_CONCEPT_ID'] = 32817

    ## Match visit_occurrence_id, visit_detail_id based on chart_time
    df_cond_occ = match_visit(df_cond_occ, 'CONDITION_OCCURRENCE_ID', df_visit_occ, df_visit_detail, on = 'chart_time')
    
    # Remove columns that aren't part of the final CONDITION_OCCURRENCE table format
    df_cond_occ.drop(columns=['subject_id', 'chart_date', 'chart_time'], inplace=True)
    df_cond_occ = df_cond_occ.astype({'CONDITION_SOURCE_CONCEPT_ID':'Int64', 'CONDITION_CONCEPT_ID':'Int64', 'VISIT_OCCURRENCE_ID':'Int64', 'VISIT_DETAIL_ID':'Int64'})

    # Save the final df_cond_occ DataFrame data to a parquet file
    df_cond_occ.to_parquet(f'{save_path}/{dname}_CONDITION_OCCURRENCE.parquet')
    if save_csv:
        df_cond_occ.to_csv(f'{save_path}/{dname}_CONDITION_OCCURRENCE.csv', index=False)
    df_cond_occ[:1000].to_csv(f'{save_path}/sample/{dname}_CONDITION_OCCURRENCE.csv', index=False)    
    print('done')


    ### DRUG EXPOSURE TABLE ###
    print('DRUG EXPOSURE TABLE...', end='')
    # Create an empty dataframe for DRUG_EXPOSURE table
    df_drug = pd.DataFrame(columns = ['DRUG_EXPOSURE_ID'])

    # Copy PERSON_ID values from the PERSON table to the DRUG_EXPOSURE table
    df_drug['PERSON_ID'] = df_person['PERSON_ID']
    # Copy PERSON_SOURCE_VALUE values as subject_id from df_person to df_drug
    df_drug['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Filter out rows in df_vitals with null 'vocab' values
    df_v = df_vitals.dropna(subset='vocab')
    # Filter rows in df_vitals where 'vocab' column contains the string 'RxNorm'
    df_v = df_v.loc[df_v['vocab'].str.contains('RxNorm')]
    # Select relevant columns and rename 'item_name' to 'drug_name'
    df_v = df_v[['subject_id', 'chart_time', 'item_name', 'value', 'concept_id']]
    df_v.rename(columns={'item_name': 'drug_name'}, inplace=True)
    # Assign a source where the data come from
    df_m['source'] = 'medi'
    # Assign a default value 'iv' to the new 'route' column
    df_v['route'] = 'iv'

    # Remove the 'concept_name' column from df_medi
    df_m = df_medi.drop(columns='concept_name')
    # Assign a source where the data come from
    df_m['source'] = 'medi'
    # Concatenate df_v and df_m vertically
    df_merge = pd.concat([df_v, df_m], axis = 0)

    # Free up memory by deleting df_v and df_m
    del df_v, df_m

    # Merge df_drug with df_merge on 'subject_id' to add details from df_merge
    df_drug = df_drug.merge(df_merge, on='subject_id', how='left')

    # Assign unique sequential IDs to the 'DRUG_EXPOSURE_ID' column
    df_drug['DRUG_EXPOSURE_ID'] = start_index['drug_exposure'] + np.arange(len(df_drug)) + 1
    # Map 'concept_id' values to 'DRUG_CONCEPT_ID' column
    df_drug['DRUG_CONCEPT_ID'] = df_drug['concept_id']

    # Define the reference date for start and end times
    base_date = datetime(2011, 1, 1)
    # Convert 'chart_time' values (in minutes) to dates using the reference base_date
    df_drug['DRUG_EXPOSURE_START_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_drug['chart_time'], unit='min')
    # Extract the date part for DRUG_EXPOSURE_START_DATE
    df_drug['DRUG_EXPOSURE_START_DATE'] = pd.to_datetime(df_drug['DRUG_EXPOSURE_START_DATETIME'].dt.date)
    # Assign the start datetime to the end datetime column (assuming no gap)
    df_drug['DRUG_EXPOSURE_END_DATETIME'] = df_drug['DRUG_EXPOSURE_START_DATETIME']
    # Assign the start date to the end date column
    df_drug['DRUG_EXPOSURE_END_DATE'] = df_drug['DRUG_EXPOSURE_START_DATE']

    # Assign the DRUG_TYPE_CONCEPT_ID indicating the data source is an EHR since it is not specified
    df_drug['DRUG_TYPE_CONCEPT_ID'] = 32817

    # Map drug quantity values from 'value' column
    df_drug['QUANTITY'] = df_drug['value']

    # Map drug administration route to corresponding Standard Concept IDs
    df_drug['ROUTE_CONCEPT_ID'] = df['route'].map({'po': 4132161, 'iv': 4171047, 'ex': 4263689}, na_action='ignore')

    ## Match visit_occurrence_id, visit_detail_id based on chart_time
    df_drug = match_visit(df_drug, 'DRUG_EXPOSURE_ID', df_visit_occ, df_visit_detail, on = 'chart_time')

    # Map drug names to the 'DRUG_SOURCE_VALUE' column
    df_drug['DRUG_SOURCE_VALUE'] = df_drug['drug_name']

    # Map drug routes to the 'ROUTE_SOURCE_VALUE' column
    df_drug['ROUTE_SOURCE_VALUE'] = df_drug['route']

    # Filter the columns in df_drug to keep only the originally defined columns
    df_drug.drop(columns=['subject_id', 'chart_time', 'value', 'concept_id', 'drug_name', 'route', 'source', 'chart_date'], inplace=True)
    df_drug = df_drug.astype({'DRUG_CONCEPT_ID': 'Int64', 'ROUTE_CONCEPT_ID': 'Int64', 'VISIT_OCCURRENCE_ID':'Int64', 'VISIT_DETAIL_ID':'Int64'})

    # Save the final df_drug DataFrame data to a parquet file
    df_drug.to_parquet(f'{save_path}/parquet/{dname}_DRUG_EXPOSURE.parquet')
    if save_csv:
        df_drug.to_csv(f'{save_path}/{dname}_DRUG_EXPOSURE.csv', index=False)
    df_drug[:1000].to_csv(f'{save_path}/sample/{dname}_DRUG_EXPOSURE.csv', index=False)    
    print('done')

    
    ### PROCEDURE_OCCURRENCE ###
    print('PROCEDURE_OCCURRENCE TABLE...', end='')
    # Create an empty dataframe for PROCEDURE_OCCURRENCE table
    df_proc = pd.DataFrame(columns=['PROCEDURE_OCCURRENCE_ID'])

    # Map corresponding PERSON_ID values from the PERSON table to the new PROCEDURE_OCCURRENCE DataFrame
    df_proc['PERSON_ID'] = df_person['PERSON_ID']

    # Extract subject_id from PERSON_SOURCE_VALUE for mapping with the operation data
    df_proc['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Merge operation data with the procedure occurrence table based on subject_id
    usecols = ['caseid', 'subject_id', 'hadm_id', 'opstart_time', 'opend_time', 'icd10_pcs', 'standard_concept_id', 'source_concept_id']
    df_proc = df_proc.merge(df_op[usecols], on='subject_id', how='left')

    # Generate unique identifiers for each procedure occurrence
    df_proc['PROCEDURE_OCCURRENCE_ID'] = start_index['procedure_occurrence'] + np.arange(len(df_proc)) + 1

    # Assign the standard concept IDs to the procedure concept ID column
    df_proc['PROCEDURE_CONCEPT_ID'] = df_proc['standard_concept_id']

    # Convert operation start and end times to datetime format using a defined base date
    base_date = datetime(2011, 1, 1)
    df_proc['PROCEDURE_DATETIME'] = base_date + pd.to_timedelta(df_proc['opstart_time'], unit='min')
    df_proc['PROCEDURE_DATE'] = df_proc['PROCEDURE_DATETIME'].dt.date
    df_proc['PROCEDURE_END_DATETIME'] = base_date + pd.to_timedelta(df_proc['opend_time'], unit='min')
    df_proc['PROCEDURE_END_DATE'] = df_proc['PROCEDURE_END_DATETIME'].dt.date

    # Assign a type concept ID indicating the data is sourced from an EHR
    df_proc['PROCEDURE_TYPE_CONCEPT_ID'] = 32817

    # Link each procedure to a corresponding visit by merging with the visit occurrence data
    df_proc['VISIT_OCCURRENCE_ID'] = df_proc.merge(df_visit_occ[['hadm_id', 'VISIT_OCCURRENCE_ID']], 
                                                on='hadm_id', suffixes=('_x', None), how='left')['VISIT_OCCURRENCE_ID']
    df_proc['VISIT_DETAIL_ID'] = df_proc.merge(df_visit_detail[['hadm_id', 'VISIT_DETAIL_ID']], 
                                                    on='hadm_id', suffixes=('_x', None), how='left')['VISIT_DETAIL_ID']

    # Populate source value and source concept ID columns using the operation data
    df_proc['PROCEDURE_SOURCE_VALUE'] = df_proc['icd10_pcs']
    df_proc['PROCEDURE_SOURCE_CONCEPT_ID'] = df_proc['source_concept_id']

    # Filter the columns in df_drug to keep only the originally defined columns
    # caseid column is left to link with VitalDB biosignal data
    df_proc.drop(columns=usecols[1:], inplace=True)
    df_proc = df_proc.astype({'PROCEDURE_CONCEPT_ID': 'Int64', 'VISIT_OCCURRENCE_ID':'Int64', 'VISIT_DETAIL_ID':'Int64'})

    # Export the final PROCEDURE_OCCURRENCE data to a parquet file
    df_proc.to_parquet(f'{save_path}/{dname}_PROCEDURE_OCCURRENCE.parquet')
    if save_csv:
        df_proc.to_csv(f'{save_path}/{dname}_PROCEDURE_OCCURRENCE.csv', index=False)
    print('done')
    df_proc[:1000].to_csv(f'{save_path}/sample/{dname}_PROCEDURE_OCCURRENCE.csv', index=False)
    
    
    # Add record_type for labs, vitals, ward table
    df_labs['record_type'] = 'PERI-OP'
    df_vitals['record_type'] = 'INTRA-OP'
    df_ward['record_type'] = 'PERI-OP'

    ### MEASUREMENT TABLE ###
    print('MEASUREMENT TABLE...', end='')
    # Create an empty dataframe for MEASUREMENT table
    df_measure = pd.DataFrame(columns=['MEASUREMENT_ID'])

    # Populate 'PERSON_ID' and 'subject_id' columns in MEASUREMENT table from the PERSON table
    df_measure['PERSON_ID'] = df_person['PERSON_ID']
    df_measure['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Filter out measurements from df_vitals where the vocabulary is not LOINC
    df_v = df_vitals.dropna(subset='vocab')
    df_v = df_v[df_v['vocab']=='LOINC'].drop(['op_id', 'vocab'], axis=1)

    # Combine data from various sources (labs, vitals, wards) into a single DataFrame
    records = pd.concat([df_labs, df_v, df_ward], axis=0)
    # Enrich the combined records with associated unit concept IDs
    records = records.merge(df_params[['Unit', 'unit_concept_id']].drop_duplicates(subset='Unit'), on='Unit', how='left')
    # Release memory
    del df_v

    # Merge the enriched records with MEASUREMENT table on 'subject_id'
    df_measure = df_measure.merge(records, on='subject_id', how='left')
    # Release memory
    del records

    # Extract and set the relevant concept and datetime details for each measurement
    df_measure['MEASUREMENT_CONCEPT_ID'] = df_measure['concept_id']
    
    base_date = datetime(2011, 1, 1)
    df_measure['MEASUREMENT_DATETIME'] = base_date + pd.to_timedelta(df_measure['chart_time'], unit='min')
    df_measure['MEASUREMENT_DATE'] = df_measure['MEASUREMENT_DATETIME'].dt.date
    # Assign 32838 (EHR Episode Record) for Intra-Op record, and 32817 (EHR) for others (Post-Op, Pre-Op).
    df_measure['MEASUREMENT_TYPE_CONCEPT_ID'] = df_measure['record_type'].map({'INTRA-OP': 32838 , 'PERI-OP': 32817})
    df_measure['OPERATOR_CONCEPT_ID'] = 4172703  # '=' operation

    # Drop rows that have non-valid meas_value (not float)
    print('Removing invalid rows...', end='')
    tlen = len(df_measure)
    df_measure['value'] = pd.to_numeric(df_measure['value'], errors='coerce')
    df_measure.dropna(subset='value', inplace=True)
    print(f'removed {tlen-len(df_measure)} rows out of {tlen} rows ...', end='')
    
    # Handle special cases for 'VALUE_AS_NUMBER' based on specific concept IDs
    # In ETL conventions, it is recommended to set the VALUE_AS_NUMBER to NULL when the value from source data is negative with the exceptions below 
    #exceptions = [3003396, 3002032, 3006277, 3012501, 3003129, 3004959, 3007435]
    #valid_mask = (df_measure['value'] >= 0) | df_measure['concept_id'].isin(exceptions)
    #df_measure.loc[valid_mask, 'VALUE_AS_NUMBER'] = df_measure['value']
    #df_measure.loc[~valid_mask, 'VALUE_AS_NUMBER'] = None
    df_measure['VALUE_AS_NUMBER'] = df_measure['value']
    
    # Assign unique MEASUREMENT_IDs to each row
    df_measure['MEASUREMENT_ID'] = start_index['measurement'] + np.arange(1, len(df_measure) + 1)

    # Set the 'UNIT_CONCEPT_ID' values
    df_measure['UNIT_CONCEPT_ID'] = df_measure['unit_concept_id']

    ## Match visit_occurrence_id, visit_detail_id based on chart_time
    df_measure = match_visit(df_measure, 'MEASRUEMENT_ID', df_visit_occ, df_visit_detail, on='chart_time') 

    # Set source value columns
    df_measure['MEASUREMENT_SOURCE_VALUE'] = df_measure['value']
    df_measure['UNIT_SOURCE_VALUE'] = df_measure['Unit']
    df_measure['VALUE_SOURCE_VALUE'] = df_measure['value']

    # Retain only the relevant columns in the final MEASUREMENT table
    df_measure.drop(columns=['subject_id', 'chart_time', 'item_name', 'value', 'Unit', 'concept_id', 'unit_concept_id', 'chart_date', 'record_type'], inplace=True)
    df_measure = df_measure.astype({'MEASUREMENT_CONCEPT_ID': 'Int64', 'UNIT_CONCEPT_ID':'Int64', 'VISIT_OCCURRENCE_ID':'Int64', 'VISIT_DETAIL_ID':'Int64'})

    # Export the final MEASUREMENT table to CSV
    df_measure.to_parquet(f'{save_path}/{dname}_MEASUREMENT.parquet')
    if save_csv:
        df_measure.to_csv(f'{save_path}/{dname}_MEASUREMENT.csv', index=False)
    df_measure[:1000].to_csv(f'{save_path}/sample/{dname}_MEASUREMENT.csv', index=False)


    ### DEATH ###
    print('DEATH TABLE...', end='')
    # Create an empty dataframe for DEATH table
    df_death = pd.DataFrame()

    # Populate 'PERSON_ID' and 'subject_id' columns in the DEATH table from the PERSON table
    df_death['PERSON_ID'] = df_person['PERSON_ID']
    df_death['subject_id'] = df_person['PERSON_SOURCE_VALUE']

    # Merge 'inhosp_death_time' from the operations (df_op) table into the DEATH table using 'subject_id'
    usecols = ['subject_id', 'inhosp_death_time']
    df_death = df_death.merge(df_op[usecols], on='subject_id', how='left')
    df_death.dropna(subset='inhosp_death_time', inplace=True)

    # Define the reference date for datetime calculations
    base_date = datetime(2011, 1, 1)

    # Convert in-hospital death times to actual datetime objects using the base_date as the reference point
    df_death['DEATH_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_death['inhosp_death_time'], unit='min')
    df_death['DEATH_DATE'] = pd.to_datetime(df_death['DEATH_DATETIME'].dt.date)

    # Set the DEATH_TYPE_CONCEPT_ID to represent data sourced from an Electronic Health Record (EHR)
    df_death['DEATH_TYPE_CONCEPT_ID'] = 32817

    # Retain only the relevant columns in the final DEATH table
    df_death.drop(columns = usecols, inplace=True)

    # Save the final df_drug DataFrame to a CSV file
    df_death.to_parquet(f'{save_path}/{dname}_DEATH.parquet')
    if save_csv:
        df_death.to_csv(f'{save_path}/{dname}_DEATH.csv', index=False)
    df_death[:1000].to_csv(f'{save_path}/sample/{dname}_DEATH.csv', index=False)
    

    ### NOTE TABLE ###
    print('NOTE TABLE...', end='')
    # Create an empty dataframe for NOTE table
    df_note = pd.DataFrame(columns=['NOTE_ID'])

    # Populate 'PERSON_ID' and 'subject_id' columns in the NOTE table from the PERSON table
    df_note['PERSON_ID'] = df_person['PERSON_ID']
    df_note['subject_id'] = df_person['PERSON_SOURCE_VALUE']


    # residual fields that are not mapped in operation table
    res_fields = ['height', 'weight', 'asa', 'emop', 'department', 'antype', 'orin_time', 'orout_time', 'anstart_time', 'anend_time', 'cpbon_time', 'cpboff_time']
    res_op = pd.melt(df_op, id_vars=['subject_id', 'opdate'], value_vars=res_fields)
    df_note = df_note.merge(res_op, on='subject_id', how='left')
    df_note.dropna(subset='value', inplace=True, ignore_index=True)

    # Assign unique sequential IDs to the 'NOTE_ID' column
    df_note['NOTE_ID'] = start_index['note'] + np.arange(len(df_note)) + 1

    base_date = datetime(2011, 1, 1)
    df_note['NOTE_DATETIME'] = base_date + pd.to_timedelta(df_note['opdate'], unit='min')
    df_note['NOTE_DATE'] = df_note['NOTE_DATETIME'].dt.date

    # Set the NOTE_TYPE_CONCEPT_ID to represent data sourced from an Electronic Health Record (EHR)
    df_note['NOTE_TYPE_CONCEPT_ID'] = 32817

    # Use the concept id  706617(Anesthesiology) or 706502(Surgical operation).
    res_ane = ['asa', 'antype','anstart_time', 'anend_time']
    df_note.loc[df_note['variable'].isin(res_ane), 'NOTE_CLASS_CONCEPT_ID'] = 706617
    df_note.loc[~df_note['variable'].isin(res_ane), 'NOTE_CLASS_CONCEPT_ID'] = 706502
    df_note['NOTE_CLASS_CONCEPT_ID'] = df_note['NOTE_CLASS_CONCEPT_ID'].astype('Int32')

    df_note['NOTE_TITLE'] = df_note['variable']
    df_note['NOTE_TEXT'] = df_note['value'].astype('str')

    # Use the concept_id 32678(UTF-8)
    df_note['ENCODING_CONCEPT_ID'] = 32678

    # Use the concept_id 4180186(English language)
    df_note['LANGUAGE_CONCEPT_ID'] = 4180186 

    # Matches visit_occurrence_id and visit_detail_id based on chart_time
    df_note = match_visit(df_note, 'NOTE_ID', df_visit_occ, df_visit_detail, on = 'opdate')

    # Retain only the relevant columns in the final NOTE table
    df_note = df_note.drop(columns=['subject_id', 'opdate', 'variable', 'value', 'chart_date'])

    # Export the final NOTE table to CSV
    df_note.to_parquet(f'{save_path}/{dname}_NOTE.parquet')
    if save_csv:
        df_note.to_csv(f'{save_path}/{dname}_NOTE.csv', index=False)
    df_note[:1000].to_csv(f'{save_path}/sample/{dname}_NOTE.csv', index=False)
    print('done')


    # Drop hadm_id column from VISIT_OCCURRENCE and VISIT_DETAIL tables
    df_visit_occ.drop(columns='hadm_id', inplace=True)
    df_visit_detail.drop(columns='hadm_id', inplace=True)
    
    print('saving visit_occurrence and visit_detail tables...', end='')
    # Save the processed data to a parquet file
    df_visit_occ.to_csv(f'{save_path}/{dname}_VISIT_OCCURRENCE.parquet')
    if save_csv:
        df_visit_occ.to_csv(f'{save_path}/{dname}_VISIT_OCCURRENCE.csv', index=False)
    df_visit_occ[:1000].to_csv(f'{save_path}/sample/{dname}_VISIT_OCCURRENCE.csv', index=False)
    
    # Save the processed VISIT_DETAIL table to a parquet file
    df_visit_detail.to_csv(f'{save_path}/{dname}_VISIT_DETAIL.parquet')
    if save_csv:
        df_visit_detail.to_csv(f'{save_path}/{dname}_VISIT_DETAIL.csv', index=False)
    df_visit_detail[:1000].to_csv(f'{save_path}/sample/{dname}_VISIT_DETAIL.csv', index=False)    
    print('done')
    

def match_visit(table, unique_id, df_visit_occ, df_visit_detail, on='chart_time'):
    # Matches visit_occurrence_id and visit_detail_id based on chart_time
    # table: a target table that needs to match visit_ids
    # unique_id: an unique identifier of a table
    # on: the column name for the reference
    
    # Convert 'chart_time' values (in minutes) again for merging with visit occurrences
    table['chart_date'] = base_date + pd.to_timedelta(table[on], unit='min')

    # Match drug exposure dates with visit occurrences based on 'PERSON_ID'
    result = pd.merge(table[['PERSON_ID', 'chart_date', unique_id]], 
                    df_visit_occ[['PERSON_ID', 'VISIT_OCCURRENCE_ID', 'VISIT_START_DATETIME', 'VISIT_END_DATETIME']], 
                      on='PERSON_ID', how='left')
    # Filter results to keep only those rows where 'chart_date' falls within a visit's start and end times
    result = result[(result['chart_date'] >= result['VISIT_START_DATETIME']) & 
                    (result['chart_date'] <= result['VISIT_END_DATETIME'])]

    
    # Merge the filtered results with df_cond_occ to add 'VISIT_OCCURRENCE_ID' details to table
    table = table.merge(result[[unique_id, 'VISIT_OCCURRENCE_ID']], 
                                    on=unique_id, 
                                    how='left', 
                                    suffixes=('_x', None))
    del result

    # Match drug exposure dates with visit occurrences based on 'PERSON_ID'
    result = pd.merge(table[['PERSON_ID', 'chart_date', unique_id]], 
                    df_visit_detail[['PERSON_ID', 'VISIT_DETAIL_ID', 'VISIT_DETAIL_START_DATETIME', 'VISIT_DETAIL_END_DATETIME']], 
                      on='PERSON_ID', how='left')
    # Filter results to keep only those rows where 'chart_date' falls within a visit's start and end times
    result = result[(result['chart_date'] >= result['VISIT_DETAIL_START_DATETIME']) & 
                    (result['chart_date'] <= result['VISIT_DETAIL_END_DATETIME'])]

    # Merge the filtered results with df_cond_occ to add 'VISIT_DETAIL_ID' details to table
    table = table.merge(result[[unique_id, 'VISIT_DETAIL_ID']], 
                                    on=unique_id, 
                                    how='left', 
                                    suffixes=('_x', None))  
    del result
    
    return table


def patient_record(subject_id):
    # This function returns every relevant records stored in OMOP-CDM tables for given subject_id
    # subject_id: an unique ID given in source table
    
    return 0
    
    
# ETL(Extract-Transform-Load) process of INSPIRE dataset to OMOP-CDM format
##### 1. Concept Mapping #####
source_tables = concept_mapping(input_path = 'inspire', mapped_path = 'inspire/mapped')
    
##### 2. Table Mapping #####
# Save the results of etl process in csv
table_mapping(source_tables, save_path = 'omop_cdm/inspire', save_csv = True)
