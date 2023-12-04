import pandas as pd
import numpy as np
from datetime import datetime
import os, time
import vitaldb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def download_vitals(save_path, multiprocess=True):
    # save_path : path to save the downloaded vital files
    # multiprocess : If True, use multiprocessing to speed up the process
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    def download_case(caseid):
        vitaldb.VitalFile(caseid).to_parquet(f'{save_path}/{caseid}.parquet')

    print('Downloading vital files', end='...')
    caseids = range(1, 6389)
    if multiprocess:
        # Use ProcessPoolExecutor to parallelize the operation
        with ProcessPoolExecutor() as executor:
            # Use tqdm for progress bar
            list(tqdm(executor.map(download_case, caseids), total=len(caseids)))
        
    else:
        for caseid in tqdm(caseids):
            download_case(caseid)
    print('done')

  
def vf_to_tables(vital_path, mapped_path, multiprocess=True):
    # Extract intraoperative measurement and drug records from vital files
    # vital_path : path of download vital files from vitaldb.net
    # mapped_path : path of mapped files of VitalDB dataset
    # multiprocess : If True, use multiprocessing to speed up the process
    
    def process_mcase(caseid):
        ipath = os.path.join(vital_path, f'{caseid}.parquet')

        # Load and extract only necessary tracks
        vf = pd.read_parquet(ipath)
        vf = vf[vf['tname'].isin(trks)]

        # Merge the standard concept of tracks
        vf = vf.merge(param_meas[['tname', 'Parameter', 'concept_id', 'Unit', 'unit_concept_id']], on='tname', how='left')
        vf['caseid'] = caseid

        vf.drop(columns='wval', inplace=True)
        vf = vf.astype({'concept_id': 'Int64', 'unit_concept_id': 'Int64'})

        return vf
    
    def process_dcase(caseid):
        ipath = os.path.join(vital_path, f'{caseid}.parquet')

        # Load and extract only neccessary tracks
        vf = pd.read_parquet(ipath)
        vf = vf[vf['tname'].isin(d_trks)]
        
        if len(vf) == 0:
            return None
            
        # Get average values for drug administration dose
        vf = vf.groupby('tname').agg({'nval': 'mean', 'dt': ['min', 'max']}).reset_index(drop=False)
        vf.columns =  [' '.join(col).strip() for col in vf.columns.values]

        # Merge the standard concept of tracks
        vf = vf.merge(param_drug[['tname', 'Parameter', 'concept_id', 'Unit', 'unit_concept_id']], on='tname', how='left')
        vf['caseid'] = caseid

        vf = vf.astype({'concept_id': 'Int64', 'unit_concept_id': 'Int64'})

        return vf
    
    # Load manually mapped parameters
    param_meas = pd.read_csv(f'{mapped_path}/vitals_vitaldb_mapped.csv')
    param_meas.dropna(subset='concept_id', inplace=True)

    # Get track names and drop duplicated rows
    params['tname'] = params['Parameter'].str.split('/').str[1]
    params.drop_duplicates(subset='tname', inplace=True)
    params.reset_index(drop=True, inplace=True)

    # Extract parameters of measurements
    param_meas = params[params['vocabulary'].isin(['LOINC', 'SNOMED'])].copy()
    # Extract intraop drugs from vital files
    param_drug = params[params['vocabulary'].str.contains('RxNorm')].copy()

    # Track names to extract
    m_trks = param_meas['tname'].values
    d_trks = param_drug['tname'].values
    
    # Get a list of all caseids
    caseids = vitaldb_info['caseid'].tolist()

    if multiprocess:
        # Use ProcessPoolExecutor to parallelize the operation
        with ProcessPoolExecutor() as executor:
            # Use tqdm for progress bar
            results_m = list(tqdm(executor.map(process_mcase, caseids), total=len(caseids)))
            results_d = list(tqdm(executor.map(process_dcase, caseids), total=len(caseids)))

        # Concatenate all DataFrames from the results
        vitaldb_meas = pd.concat(results_m, axis=0)
        vitaldb_drugs = pd.concat(results_d, axis=0)
        
    else:
        # Define measurement, drug tables for vitaldb
        vitaldb_meas = pd.DataFrame(columns=['caseid'])
        vitaldb_drugs = pd.DataFrame(columns=['caseid'])
        
        for caseid in caseids:
            vitaldb_meas = pd.concat([vitaldb_meas, process_case(caseid)], axis=0)
            vitaldb_drugs = pd.concat([vitaldb_drugs, process_dcase(caseid)], axis=0)

    # Drop columns with no valid value
    vitaldb_meas.dropna(subset=['nval'], inplace=True)
    vitaldb_drugs = vitaldb_drugs[vitaldb_drugs['nval mean']!=0]
    
    # Unify field names with labs table
    vitaldb_meas = vitaldb_meas.drop(columns='Parameter')
    vitaldb_meas.rename(columns={'tname': 'name', 'nval': 'result'}, inplace=True)
    
    vitaldb_meas.reset_index(drop=True, inplace=True)
    vitaldb_drugs.reset_index(drop=True, inplace=True)

    return vitaldb_meas, vitaldb_drugs


def preprocessing(vital_path, mapped_path, inspire_path):
    '''
    Preprocessing stage of vitaldb dataset.
    vital_path : path to save vital files of VitalDB dataset
    mapped_path : path for mapped files of VitalDB dataset. It includes mapping of opname, hadm_id.
    inspire_path : path for INSPIRE dataset to access operations table
    '''
    
    # Load clinical information table and labs table from api.vitaldb
    #vitaldb_info = pd.read_csv('https://api.vitaldb.net/cases')
    vitaldb_labs = pd.read_csv('https://api.vitaldb.net/labs')
    
    # Load mapped clinical information table (opname, hadm_id)
    vitaldb_info = pd.read_csv(f'{mapped_path}/vitaldb_info_mapped.csv')
    
    # Drop duplicates rows of labs table
    vitaldb_labs.drop_duplicates(inplace=True)

    # Load mapped parameters of vitaldb dataset
    vitaldb_params = pd.read_csv(f'{mapped_path}/parameters_vitaldb_mapped.csv')
    vitaldb_params.dropna(subset='concept_id', inplace=True)
    vitaldb_params = vitaldb_params.astype({'concept_id': 'Int64', 'unit_concept_id': 'Int64'})

    # Map labs table with vitaldb_params table
    vitaldb_labs = vitaldb_labs.merge(vitaldb_params[['Label', 'concept_id', 'Unit', 'unit_concept_id']], left_on='name', right_on='Label', how='left')
    vitaldb_labs.drop(columns='Label', inplace=True)
    vitaldb_labs = vitaldb_labs.astype({'concept_id': 'Int64', 'unit_concept_id': 'Int64'})
    
    ## Remove caseids that overlap with INSPIRE dataset
    # Load operations table from INSPIRE dataset
    df_op = pd_read_csv(f'{inspire_path}/operations.csv')
    
    ## Extract caseid that are not included in INSPIRE
    vitaldb_info = vitaldb_info[vitaldb_info['caseid'].isin(df_op['case_id'].unique())]

    # Whether to distinguish multiple surgeries(caseid) for subjectid
    distinct_caseid = False

    if not distinct_caseid:
        # Select one caseid for subjectid with multiple caseids
        vitaldb_info = vitaldb_info.drop_duplicates(subset=['subjectid'], keep='first')

    else:
        ## Add lab info to info table
        # Extract all labs(hct, na) records for each caseid
        def join_str(x):
            return ','.join(x.astype(str))
        #hcts = vitaldb_labs[vitaldb_labs['name'] == 'hct'].groupby('caseid')['result'].apply(','.join).reset_index()
        hcts = vitaldb_labs[vitaldb_labs['name'] == 'hct'].groupby('caseid').agg({'result': join_str, 'dt': join_str}).reset_index()
        hcts.rename(columns={'result': 'hcts', 'dt':'hct_dt'}, inplace=True)

        #nas = vitaldb_labs[vitaldb_labs['name'] == 'na'].groupby('caseid')['result'].apply(','.join).reset_index()
        nas = vitaldb_labs[vitaldb_labs['name'] == 'na'].groupby('caseid').agg({'result': join_str, 'dt': join_str}).reset_index()
        nas.rename(columns={'result': 'nas', 'dt': 'nas_dt'}, inplace=True)

        # merge hct values
        vitaldb_info = vitaldb_info.merge(hcts, on='caseid', how='left')
        vitaldb_info['hcts_cut'] = vitaldb_info['hcts'].str[:15]
        vitaldb_info = vitaldb_info.merge(nas, on='caseid', how='left')
        vitaldb_info['nas_cut'] = vitaldb_info['nas'].str[:15]
        

    ## Assign hadm_id
    # subjectid that have more than one caseid
    subjects = vitaldb_info.groupby('subjectid').agg({'caseid': 'count'})
    unique_subjects = subjects[subjects['caseid'] > 1].index

    # Assign a unique value for hadm_id which subjectid only have one caseid (하나의 caseid만 있는 subjectid의 hamd_id 부여)
    mask = ~vitaldb_info['subjectid'].isin(unique_subjects)
    vitaldb_info.loc[mask, 'hadm_id'] = np.arange(mask.sum()) + 1

    # Assign a base_time to discriminate the base time of caseids in same subjectid
    vitaldb_info.loc[mask, 'base_time'] = 0

    vitaldb_info = vitaldb_info.astype({'hadm_id': 'Int64', 'base_time': 'Int64'})

    ## Get min, max time for each caseid
    # Extract min, max value of dt for each caseid
    df_dts = vitaldb_labs.groupby('caseid').agg({'dt':[min, max]}).reset_index(drop=False)
    # Flatten the MultiIndex for columns
    df_dts.columns = [' '.join(col).strip() for col in df_dts.columns.values]

    vitaldb_info = vitaldb_info.merge(df_dts, on='caseid', how='left') 

    vitaldb_info['max_time'] = vitaldb_info[['dis', 'dt max']].max(axis=1)
    vitaldb_info['min_time'] = vitaldb_info[['adm', 'dt min']].min(axis=1)
    
    # Map intraoperative drugs from clinical information table
    info_params = vitaldb_params[vitaldb_params['vocab'].str.contains('RxNorm')]
    vitaldb_drugs = pd.melt(vitaldb_info, id_vars=['caseid', 'hadm_id', 'subjectid', 'opstart', 'opend', 'base_time'], value_vars= info_params['Label'])
    vitaldb_drugs['chart_time'] = (vitaldb_drugs['opstart'] + vitaldb_drugs['opend']) / 2
    vitaldb_drugs = vitaldb_drugs.merge(info_params[['Label', 'concept_id', 'Unit', 'unit_concept_id']], left_on='variable', right_on='Label', how='left')
    
    # Drop rows of missing value or zero value
    vitaldb_drugs = vitaldb_drugs.dropna(subset='value')
    vitaldb_drugs = vitaldb_drugs[vitaldb_drugs['value']!=0]
        
    ### Get measurements from Vital files
    # Download Vital files
    download_vitals(vital_path)
    
    # Convert vital files to measurements table
    vitaldb_meas, vf_drugs = vf_to_tables(vital_path, mapped_path)
    
    source_tables = {'information': vitaldb_info, 'labs': vitaldb_labs, 'drugs': vitaldb_drugs, 'measurements': vitaldb_meas, 'vf_drugs': vf_drugs}
    
    return source_tables


def table_mapping(source_tables, save_path, save_csv=False):
    '''
    Map source tables to OMOP-CDM standardized tables. 
    Saves the mapped cdm tables in parquet.
    save_path : path to save the mapped cdm tables
    save_csv : If True, also save the cdm tables into csv.
    '''
    # Make directory to save the results
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    os.makedirs(os.path.join(save_path, 'sample'), exist_ok=True)
    
    # Load source tables
    vitaldb_info = source_tables['information']
    vitaldb_labs = source_tables['labs']
    vitaldb_drugs = source_tables['drugs']
    vitaldb_meas = source_tables['measurements']
    vf_drugs = source_tables['vf_drugs']
    
    # start_index for each table_id
    start_index = {
    'person': 1000000,
    'observation_period': 1000000,
    'visit_occurrence': 1000000,
    'visit_detail': 1000000,
    'condition_occurrence': 1000000,
    'drug_exposure': 1000000,
    'procedure_occurrence': 1000000,
    'measurement': 1000000,
    'note': 1000000,
    'location': 1000000 
    }
    # name of source dataset
    dname = 'VITALDB'
    
    ### PERSON TABLE ###
    print('PERSON TABLE...', end='')
    # Create an empty dataframe for PERSON table
    df_person = pd.DataFrame(columns=['PERSON_ID'])
    
    # Assign unique IDs to each distinct 'subjectid' from the operations data
    unique_ids = vitaldb_info['subjectid'].unique()
    df_person['PERSON_ID'] = start_index['person'] + np.arange(len(unique_ids)) + 1
    df_person['subjectid'] = unique_ids

    # Merge relevant columns from the operations dataframe with the PERSON dataframe based on 'subject_id'
    usecols = ['subjectid', 'age', 'sex']
    df_person = df_person.merge(vitaldb_info[usecols], on = 'subjectid')
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

    # Assign value for LOCATION_ID (1: Vitaldb)
    df_person['LOCATION_ID'] = 'vitaldb'

    # Populate source value columns based on values from the operations data
    df_person['PERSON_SOURCE_VALUE'] = df_person['subjectid']
    df_person['GENDER_SOURCE_VALUE'] = df_person['sex']

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
    df_obs['subjectid'] = df_person['PERSON_SOURCE_VALUE']
    
    # Merge min_time, max_time, base_time from information table
    usecols = ['subjectid', 'min_time', 'max_time', 'base_time']
    df_obs = df_obs.merge(vitaldb_info[usecols], on='subjectid', how='left')
    
    # Define the base date for the observation period
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_obs['base_time'], unit='sec')
    
    # Set the OBSERVATION_PERIOD_START_DATE, OBSERVATION_PERIOD_END_DATE to the earliest and latest record of subjectid 
    df_obs['OBSERVATION_PERIOD_START_DATE'] = pd.to_datetime(base_date) + pd.to_timedelta(df_obs['min_time'], unit='sec')
    df_obs['OBSERVATION_PERIOD_END_DATE'] = pd.to_datetime(base_date) + pd.to_timedelta(df_obs['max_time'], unit='sec')

    # Assign the PERIOD_TYPE_CONCEPT_ID indicating the data source is an EHR
    df_obs['PERIOD_TYPE_CONCEPT_ID'] = 32817
    
    # Remove columns that aren't part of the final OBSERVATION_PERIOD table format
    df_obs.drop(columns=usecols, inplace=True)   
    
    # Write the processed data to a parquet file
    df_obs.to_parquet(f'{save_path}/{dname}_OBSERVATION_PERIOD.parquet')
    if save_csv:
        df_obs.to_csv(f'{save_path}/{dname}_OBSERVATION_PERIOD.csv', index=False)
    df_obs[:1000].to_csv(f'{save_path}/sample/{dname}_OBSERVATION_PERIOD.csv', index=False)    
    print('done')
    
    
    ### VISIT_OCCURRENCE TABLE ###
    # Match admission records
    print('VISIT_OCCURRENCE TABLE...', end='')
    # Create an empty dataframe for VISIT_OCCURRENCE table
    df_visit_occ = pd.DataFrame(columns=['VISIT_OCCURRENCE_ID'])

    # Copy PERSON_ID, PERSON_SOURCE_VALUE from df_person to df_visit_occ
    df_visit_occ['PERSON_ID'] = df_person['PERSON_ID']
    df_visit_occ['subjectid'] = df_person['PERSON_SOURCE_VALUE']

    # Merge admission and discharge time to VISIT_OCCURRENCE table
    df_visit_occ = df_visit_occ.merge(vitaldb_info[['subjectid', 'hadm_id', 'adm', 'dis', 'base_time']], on='subjectid', how='left')

    # Assign sequential IDs starting from 1 to VISIT_OCCURRENCE_ID column
    df_visit_occ['VISIT_OCCURRENCE_ID'] = start_index['visit_occurrence'] + np.arange(len(df_visit_occ)) + 1

    # Set VISIT_CONCEPT_ID to indicate all individuals are admitted to hospital
    df_visit_occ['VISIT_CONCEPT_ID'] = 9201

    # Define the base date
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_visit_occ['base_time'], unit='sec')
    # Set VISIT_START_DATE to the admission time
    df_visit_occ['VISIT_START_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_visit_occ['adm'], unit='sec')
    df_visit_occ['VISIT_START_DATE'] = pd.to_datetime(df_visit_occ['VISIT_START_DATETIME'].dt.date)
    # Set VISIT_END_DATE to the discharge time
    df_visit_occ['VISIT_END_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_visit_occ['dis'], unit='sec')
    df_visit_occ['VISIT_END_DATE'] = pd.to_datetime(df_visit_occ['VISIT_END_DATETIME'].dt.date)

    # Assign the VISIT_TYPE_CONCEPT_ID indicating the data source is an EHR
    df_visit_occ['VISIT_TYPE_CONCEPT_ID'] = 32817

    ## Mapping PRECEDING_VISIT_OCCURRENCE_ID
    # Generate a column with the previous 'subject_id' for determining preceding visit occurrence
    df_visit_occ['prev_subjectid'] = df_visit_occ['subjectid'].shift(1).astype('Int64')
    # Create a new boolean column 'nadm' to check if the current row's subject_id matches the previous one
    df_visit_occ['nadm'] = df_visit_occ['subjectid'] == df_visit_occ['prev_subjectid']
    # Set the first row's 'nadm' value to False since there's no preceding record
    df_visit_occ.at[0, 'nadm'] = False 
    # Compute PRECEDING_VISIT_OCCURRENCE_ID based on 'nadm'
    df_visit_occ['PRECEDING_VISIT_OCCURRENCE_ID'] = np.where(df_visit_occ['nadm'], df_visit_occ['VISIT_OCCURRENCE_ID'].shift(1), np.nan)
    df_visit_occ['PRECEDING_VISIT_OCCURRENCE_ID'] = df_visit_occ['PRECEDING_VISIT_OCCURRENCE_ID'].astype('Int64')

    # Remove columns that aren't part of the final VISIT_OCCURRENCE table format except for 'hadm_id'
    df_visit_occ.drop(columns=['subjectid', 'prev_subjectid', 'nadm', 'adm', 'dis', 'base_time'], inplace=True)

    
    ### VISIT_DETAIL TABLE ###
    # Match ICU_ADMIN record
    # Since there is no icu_in, icu_out time, we use hospital admission, discharge time as visit detail start, end time.
    # Create a new DataFrame for VISIT_DETAIL data with the specified columns
    df_visit_detail = pd.DataFrame(columns=['VISIT_DETAIL_ID'])

    # Populate the PERSON_ID and MRN columns with data from the df_person DataFrame
    df_visit_detail['PERSON_ID'] = df_person['PERSON_ID']
    df_visit_detail['subjectid'] = df_person['PERSON_SOURCE_VALUE']

    # Integrate visit detail data with operational data from df_op using 'MRN'
    usecols = ['hadm_id', 'subjectid', 'adm', 'dis', 'icu_days', 'base_time']
    df_visit_detail = df_visit_detail.merge(vitaldb_info[usecols], on='subjectid')

    # Remove rows that do not have ICU record
    df_visit_detail = df_visit_detail[df_visit_detail['icu_days']==0].reset_index(drop=True)

    # Generate unique sequential IDs for VISIT_DETAIL_ID
    df_visit_detail['VISIT_DETAIL_ID'] = start_index['visit_detail'] + np.arange(len(df_visit_detail)) + 1

    # Designate a concept ID representing ICU visits
    df_visit_detail['VISIT_DETAIL_CONCEPT_ID'] = 32037

    # Calculate visit start and end datetime values using base_date and ICU admission/discharge times
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_visit_detail['base_time'], unit='sec')
    # Assign visit start and end datetime values to hosptial admission, discharge times
    df_visit_detail['VISIT_DETAIL_START_DATETIME'] = base_date + pd.to_timedelta(df_visit_detail['adm'], unit='sec')
    df_visit_detail['VISIT_DETAIL_START_DATE'] = df_visit_detail['VISIT_DETAIL_START_DATETIME'].dt.date
    df_visit_detail['VISIT_DETAIL_END_DATETIME'] = base_date + pd.to_timedelta(df_visit_detail['dis'], unit='sec')
    df_visit_detail['VISIT_DETAIL_END_DATE'] = df_visit_detail['VISIT_DETAIL_END_DATETIME'].dt.date

    # Specify the concept ID for the visit detail type as sourced from EHR
    df_visit_detail['VISIT_DETAIL_TYPE_CONCEPT_ID'] = 32817

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
    df_cond_occ['subjectid'] = df_person['PERSON_SOURCE_VALUE']

    # Merge the diagnosis information from information table
    usecols = ['subjectid', 'hadm_id', 'dx', 'adm', 'dis', 'base_time']
    df_cond_occ = df_cond_occ.merge(vitaldb_info[usecols], on='subjectid')

    # Assign sequential IDs starting from 1 to CONDITION_OCCURRENCE_ID column
    df_cond_occ['CONDITION_OCCURRENCE_ID'] = start_index['condition_occurrence'] + np.arange(len(df_cond_occ)) + 1

    # Define the base date
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_cond_occ['base_time'], unit='sec')
    # Set condition_start_date as hospital admission time since there is no condition start_time.
    df_cond_occ['CONDITION_START_DATETIME'] = base_date + pd.to_timedelta(df_cond_occ['adm'], unit='sec')
    df_cond_occ['CONDITION_START_DATE'] = pd.to_datetime(df_cond_occ['CONDITION_START_DATETIME'].dt.date)

    # Set condition_end_date as hospital discharge time since there is no condition_end_time.
    df_cond_occ['CONDITION_END_DATETIME'] = base_date + pd.to_timedelta(df_cond_occ['dis'], unit='sec')
    df_cond_occ['CONDITION_END_DATE'] = pd.to_datetime(df_cond_occ['CONDITION_END_DATETIME'].dt.date)

    # Assign the CONDITION_TYPE_CONCEPT_ID indicating the data source is an EHR
    df_cond_occ['CONDITION_TYPE_CONCEPT_ID'] = 32817

    ## Match visit_occurrence_id, visit_detail based on hadm_id
    # It is possible since time span of visit_occ, visit_detail id are same with hadm_id 
    df_cond_occ = df_cond_occ.merge(df_visit_occ[['hadm_id', 'VISIT_OCCURRENCE_ID']], on='hadm_id', how='left')
    df_cond_occ = df_cond_occ.merge(df_visit_detail[['hadm_id', 'VISIT_DETAIL_ID']], on='hadm_id', how='left')

    # Remove columns that aren't part of the CONDITION_OCCURRENCE table format
    df_cond_occ.drop(columns=usecols, inplace=True)
    df_cond_occ = df_cond_occ.astype({'VISIT_OCCURRENCE_ID':'Int64', 'VISIT_DETAIL_ID':'Int64'})

    # Save the final df_cond_occ DataFrame to a parqeut file
    df_cond_occ.to_parquet(f'{save_path}/{dname}_CONDITION_OCCURRENCE.parquet')
    if save_csv:
        df_cond_occ.to_csv(f'{save_path}/{dname}_CONDITION_OCCURRENCE.csv', index=False)
    print('done')


    ### DRUG_EXPOSURE TABLE ###
    print('DRUG_EXPOSURE TABLE...', end='')
    # Create an empty DataFrame using the predefined column names
    df_drug = pd.DataFrame(columns = ['DRUG_EXPOSURE_ID'])

    # Copy PERSON_ID values from the PERSON table to the DRUG_EXPOSURE table
    df_drug['PERSON_ID'] = df_person['PERSON_ID']
    # Copy PERSON_SOURCE_VALUE values as MRN from df_person to df_drug
    df_drug['subjectid'] = df_person['PERSON_SOURCE_VALUE']

    # Merge the drug information from source drugs table
    usecols = ['subjectid', 'hadm_id', 'value', 'opstart', 'opend', 'Label', 'concept_id', 'Unit', 'unit_concept_id', 'base_time']
    df_drug = df_drug.merge(vitaldb_drugs[usecols], on='subjectid', how='left')

    # Assign unique sequential IDs to the 'DRUG_EXPOSURE_ID' column
    df_drug['DRUG_EXPOSURE_ID'] = start_index['drug_exposure'] + np.arange(len(df_drug)) + 1
    # Map 'concept_id' values to 'DRUG_CONCEPT_ID' column
    df_drug['DRUG_CONCEPT_ID'] = df_drug['concept_id']

    # Define the reference date for start and end times
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_drug['base_time'], unit='sec')
    # Convert 'opstart' values (in seconds) to dates using the reference base_date
    df_drug['DRUG_EXPOSURE_START_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_drug['opstart'], unit='sec')
    # Extract the date part for DRUG_EXPOSURE_START_DATE
    df_drug['DRUG_EXPOSURE_START_DATE'] = pd.to_datetime(df_drug['DRUG_EXPOSURE_START_DATETIME'].dt.date)
    # Assign the start datetime to the end datetime column (assuming no gap)
    df_drug['DRUG_EXPOSURE_END_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_drug['opend'], unit='sec')
    # Assign the start date to the end date column
    df_drug['DRUG_EXPOSURE_END_DATE'] = pd.to_datetime(df_drug['DRUG_EXPOSURE_END_DATETIME'].dt.date)

    # Assign 32838 (EHR Episode REcord) for Intra-OP record
    df_drug['DRUG_TYPE_CONCEPT_ID'] = 32838
    
    # Map drug quantity values from 'value' column
    df_drug['QUANTITY'] = df_drug['value']
    # Map drug administration route to intravenous route
    df_drug['ROUTE_CONCEPT_ID'] = 4171047

    ## Match visit_occurrence_id, visit_detail based on hadm_id
    df_drug = df_drug.merge(df_visit_occ[['hadm_id', 'VISIT_OCCURRENCE_ID']], on='hadm_id', how='left')
    df_drug = df_drug.merge(df_visit_detail[['hadm_id', 'VISIT_DETAIL_ID']], on='hadm_id', how='left')

    # Map drug names to the 'DRUG_SOURCE_VALUE' column
    df_drug['DRUG_SOURCE_VALUE'] = df_drug['Label']
    
    # Map drug dose units
    df_drug['DOSE_UNIT_CONCEPT_ID'] = df_drug['unit_concept_id'].astype('Int64')
    df_drug['DOSE_UNIT_SOURCE_VALUE'] = df_drug['Unit']
    
    # Remove columns that aren't part of the DRUG_EXPOSURE table format
    df_drug.drop(columns=usecols, inplace=True)
    df_drug = df_drug.astype({'DRUG_CONCEPT_ID': 'Int64', 'DRUG_TYPE_CONCEPT_ID': 'Int64', 'ROUTE_CONCEPT_ID': 'Int64', 'VISIT_OCCURRENCE_ID':'Int64', 'VISIT_DETAIL_ID':'Int64'})

    # Save the final df_drug DataFrame to a parqeut file
    df_drug.to_csv(f'{save_path}/{dname}_DRUG_EXPOSURE.csv', index=False)
    df_drug.to_parquet(f'{save_path}/{dname}_DRUG_EXPOSURE.parquet')    


    ### PROCEDURE_OCCURRENCE_TABLE ###
    # Add approcah to opname
    df_proc['procedure_nm'] = df_proc['opname'] + '_' + df_proc['approach']


    ### MEASUREMENT TABLE ###
    print('MEASUREMENT TABLE...', end='') 
    # Create an empty DataFrame using the predefined column names
    df_measure = pd.DataFrame(columns = ['MEASUREMENT_ID'])

    # Copy PERSON_ID values from the PERSON table to the MEASUREMENT table
    df_measure['PERSON_ID'] = df_person['PERSON_ID']
    # Copy PERSON_SOURCE_VALUE values as MRN from df_person to df_measure
    df_measure['subjectid'] = df_person['PERSON_SOURCE_VALUE']
    
    # Merge caseid from information table
    usecols = ['caseid', 'subjectid', 'hadm_id', 'base_time']
    df_measure = df_measure.merge(vitaldb_info[usecols], on='subjectid', how='left')
    
    ## Get measurement data from vital files
    #vitaldb_meas['record_type'] = 'INTRA-OP'
    vf_measure = df_measure.merge(vitaldb_meas, on='caseid', how='left')
    
    ## Get measurement data from labs table
    #vitaldb_labs['record_type'] = 'PERI-OP'
    labs_measure = df_measure.merge(vitaldb_labs, on='caseid', how='left')
    
    # Concat measurement data from vital files and labs table
    df_measure = pd.concat([vf_measure, labs_measure], axis=0)
    df_measure.reset_index(drop=True, inplace=True)
    
    # Extract and set the relevant concept and datetime details for each measurement
    df_measure['MEASUREMENT_CONCEPT_ID'] = df_measure['concept_id']
    
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_measure['base_time'], unit='sec')
    df_measure['MEASUREMENT_DATETIME'] = base_date + pd.to_timedelta(df_measure['dt'], unit='sec')
    df_measure['MEASUREMENT_DATE'] = df_measure['MEASUREMENT_DATETIME'].dt.date
    # Assign 32838 (EHR Episode Record) for Intra-Op record, and 32817 (EHR) for others (Post-Op, Pre-Op).
    df_measure['MEASUREMENT_TYPE_CONCEPT_ID'] = 32817 #df_measure['record_type'].map({'INTRA-OP': 32838 , 'PERI-OP': 32817})
    df_measure['OPERATOR_CONCEPT_ID'] = 4172703  # '=' operation
    
    # Drop rows that have non-valid meas_value (not float)
    print('Removing invalid rows...', end='')
    tlen = len(df_measure)
    df_measure['result'] = pd.to_numeric(df_measure['result'], errors='coerce')
    df_measure.dropna(subset='result', inplace=True)
    print(f'removed {tlen-len(df_measure)} rows out of {tlen} rows ...', end='')
    
    # Handle special cases for 'VALUE_AS_NUMBER' based on specific concept IDs
    # In ETL conventions, it is recommended to set the VALUE_AS_NUMBER to NULL when the value from source data is negative with the exceptions below 
    #exceptions = [3003396, 3002032, 3006277, 3012501, 3003129, 3004959, 3007435]
    #valid_mask = (df_measure['result'] >= 0) | df_measure['concept_id'].isin(exceptions)
    #df_measure.loc[valid_mask, 'VALUE_AS_NUMBER'] = df_measure['result']
    #df_measure.loc[~valid_mask, 'VALUE_AS_NUMBER'] = None
    df_measure['VALUE_AS_NUMBER'] = df_measure['result']

    # Assign unique MEASUREMENT_IDs to each row
    df_measure['MEASUREMENT_ID'] = start_index['measurement'] + np.arange(1, len(df_measure) + 1)

    # Set the 'UNIT_CONCEPT_ID' values
    df_measure['UNIT_CONCEPT_ID'] = df_measure['unit_concept_id']
    
    ## Match visit_occurrence_id, visit_detail based on hadm_id
    df_measure = df_measure.merge(df_visit_occ[['hadm_id', 'VISIT_OCCURRENCE_ID']], on='hadm_id', how='left')
    df_measure = df_measure.merge(df_visit_detail[['hadm_id', 'VISIT_DETAIL_ID']], on='hadm_id', how='left')
    
    # Retain only the relevant columns in the final MEASUREMENT table
    df_measure.drop(columns=usecols + list(vitaldb_labs.columns), inplace=True)
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

    # Populate 'PERSON_ID' and 'subjectid' columns in the DEATH table from the PERSON table
    df_death['PERSON_ID'] = df_person['PERSON_ID']
    df_death['subjectid'] = df_person['PERSON_SOURCE_VALUE']
    
    # Merge 'inhosp_death_time' from the operations (df_op) table into the DEATH table using 'subject_id'
    usecols = ['subjectid', 'death_inhosp', 'dis', 'base_time']
    df_death = df_death.merge(vitaldb_info[usecols], on='subjectid', how='left')
    df_death = df_death[df_death['death_inhosp']==1]
    
    # Define the reference date for datetime calculations
    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_death['base_time'], unit='sec')

    # Convert in-hospital death times to actual datetime objects using the base_date as the reference point
    df_death['DEATH_DATETIME'] = pd.to_datetime(base_date) + pd.to_timedelta(df_death['dis'], unit='sec')
    df_death['DEATH_DATE'] = pd.to_datetime(df_death['DEATH_DATETIME'].dt.date)

    # Set the DEATH_TYPE_CONCEPT_ID to represent data sourced from an Electronic Health Record (EHR)
    df_death['DEATH_TYPE_CONCEPT_ID'] = 32817

    # Retain only the relevant columns in the final DEATH table
    df_death.drop(columns = usecols, inplace=True)
    df_death.reset_index(drop=True, inplace=True)

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
    df_note['subjectid'] = df_person['PERSON_SOURCE_VALUE']


    # residual fields that are not mapped in operation table
    res_fields = ['height', 'weight', 'asa', 'emop', 'department', 'ane_type', 'anestart', 'aneend']
    res_op = pd.melt(vitaldb_info, id_vars=['subjectid', 'hadm_id', 'opstart', 'base_time'], value_vars=res_fields)
    df_note = df_note.merge(res_op, on='subjectid', how='left')
    df_note.dropna(subset='value', inplace=True, ignore_index=True)

    # Assign unique sequential IDs to the 'NOTE_ID' column
    df_note['NOTE_ID'] = start_index['note'] + np.arange(len(df_note)) + 1

    base_date = datetime(2011, 1, 1) + pd.to_timedelta(df_note['base_time'], unit='sec')
    df_note['NOTE_DATETIME'] = base_date + pd.to_timedelta(df_note['opstart'], unit='sec')
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
    df_note = df_note.merge(df_visit_occ[['hadm_id', 'VISIT_OCCURRENCE_ID']], on='hadm_id', how='left')
    df_note = df_note.merge(df_visit_detail[['hadm_id', 'VISIT_DETAIL_ID']], on='hadm_id', how='left')

    # Retain only the relevant columns in the final NOTE table
    df_note = df_note.drop(columns=['subjectid', 'hadm_id', 'opstart', 'variable', 'value', 'base_time'])

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
    
    
# ETL(Extract-Transform-Load) process of INSPIRE dataset to OMOP-CDM format
##### 1. Concept Mapping #####
source_tables = preprocessing(vital_path = 'vitaldb/vital', mapped_path = 'vitaldb/mapped', inspire_path = 'inspire')
    
##### 2. Table Mapping #####
# Save the results of etl process in csv
table_mapping(source_tables, save_path = 'omop_cdm/vitaldb', save_csv = True)
