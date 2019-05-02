import os
import pandas as pd
import numpy as np
import glob
import datetime
import socket
from sklearn.preprocessing import LabelEncoder
from model import Config
from fns_categorical_encoding import preprocess_data
from constants import sample_seed

def get_data_folder():
    """
    hostname = socket.gethostname()
    if hostname in ['drago', 'drago2', 'drago3']:
        data_folder = '/storage/store/work/pcerda/data'
    elif hostname in ['paradox', 'paradigm']:
        data_folder = '/storage/local/pcerda/data'
    else:
        data_folder = os.path.join(CE_HOME, 'data')
    return data_folder
    """
    return "./"


def create_folder(path, folder):
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))
    return


def print_unique_values(df):
    for col in df.columns:
        print(col, df[col].unique().shape)
        print(df[col].unique())
        print('\n')


def check_nan_percentage(df, col_name):
    threshold = .15
    missing_fraction = df[col_name].isnull().sum()/df.shape[0]
    if missing_fraction > threshold:
        raise Exception("Fraction of missing values for column '%s'"
                        "(%.3f) is higher than %.3f"
                        % (col_name, missing_fraction, threshold))


class Data:
    def __init__(self, name):
        self.name = name
        self.configs = None
        self.xcols, self.ycol = None, None

        ''' Given the dataset name, return the respective dataframe as well as
        the the action for each column.'''

        if name in ['adult', 'adult2', 'adult3']:
            '''Source: https://archive.ics.uci.edu/ml/datasets/adult'''
            data_path = os.path.join(get_data_folder(), 'adult_dataset')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'adult.data')

        if name == 'beer_reviews':
            '''Source: BigML'''
            data_path = os.path.join(get_data_folder(), 'bigml/beer_reviews/')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'beer_reviews.csv')

        if name == 'midwest_survey':
            '''FiveThirtyEight Midwest Survey
            Original source: https://github.com/fivethirtyeight/data/tree/
                             master/region-survey
            Source: BigML'''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/FiveThirtyEight_Midwest_Survey')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'FiveThirtyEight_Midwest_Survey.csv')

        if name == 'indultos_espana':
            '''Source: '''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/Indultos_en_Espana_1996-2013')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Indultos_en_Espana_1996-2013.csv')

        if name == 'docs_payments':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'docs_payments')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'output', 'DfD.h5')

        if name == 'medical_charge':
            '''Source: BigML'''
            data_path = os.path.join(get_data_folder(),
                                     'bigml/MedicalProviderChargeInpatient')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'MedicalProviderChargeInpatient.csv')

        if name == 'road_safety':
            '''Source: https://data.gov.uk/dataset/road-accidents-safety-data
            '''
            data_path = os.path.join(get_data_folder(), 'road_safety')
            create_folder(data_path, 'output/results')
            data_file = [os.path.join(data_path, 'raw', '2015_Make_Model.csv'),
                         os.path.join(data_path, 'raw', 'Accidents_2015.csv'),
                         os.path.join(data_path, 'raw', 'Casualties_2015.csv'),
                         os.path.join(data_path, 'raw', 'Vehicles_2015.csv')]

        if name == 'consumer_complaints':
            '''Source: https://catalog.data.gov/dataset/
                       consumer-complaint-database
               Documentation: https://cfpb.github.io/api/ccdb//fields.html'''
            data_path = os.path.join(get_data_folder(), 'consumer_complaints')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Consumer_Complaints.csv')

        if name == 'traffic_violations':
            '''Source: https://catalog.data.gov/dataset/
                       traffic-violations-56dda
               Source2: https://data.montgomerycountymd.gov/Public-Safety/
                        Traffic-Violations/4mse-ku6q'''
            data_path = os.path.join(get_data_folder(), 'traffic_violations')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Traffic_Violations.csv')

        if name == 'crime_data':
            '''Source: https://catalog.data.gov/dataset/
                       crime-data-from-2010-to-present
               Source2: https://data.lacity.org/A-Safe-City/
                        Crime-Data-from-2010-to-Present/y8tr-7khq'''
            data_path = os.path.join(get_data_folder(), 'crime_data')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Crime_Data_from_2010_to_Present.csv')

        if name == 'employee_salaries':
            '''Source: https://catalog.data.gov/dataset/
                       employee-salaries-2016'''
            data_path = os.path.join(get_data_folder(), 'employee_salaries')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'Employee_Salaries_-_2016.csv')
            #
            data_file = os.path.join('Employee_Salaries_-_2016.csv')

        if name == 'product_relevance':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'product_relevance')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'train.csv')

        if name == 'federal_election':
            '''Source: https://classic.fec.gov/finance/disclosure/
                       ftpdet.shtml#a2011_2012'''
            data_path = os.path.join(get_data_folder(), 'federal_election')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'itcont.txt')
            self.data_dict_file = os.path.join(data_path, 'data_dict.csv')

        if name == 'public_procurement':
            '''Source: https://data.europa.eu/euodp/en/data/dataset/ted-csv'''
            data_path = os.path.join(get_data_folder(), 'public_procurement')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw',
                                     'TED_CAN_2015.csv')

        # add here the path to a new dataset ##################################
        if name == 'new_dataset':
            '''Source: '''
            data_path = os.path.join(get_data_folder(), 'new_dataset')
            create_folder(data_path, 'output/results')
            data_file = os.path.join(data_path, 'raw', 'data_file.csv')
        #######################################################################

        self.file = data_file
        self.path = data_path

    def preprocess(self, n_rows=-1, str_preprocess=True,
                   clf_type='regression'):
        self.col_action = {k: v for k, v in self.col_action.items()
                           if v != 'del'}
        self.xcols = [key for key in self.col_action
                      if self.col_action[key] is not 'y']
        self.ycol = [key for key in self.col_action
                     if self.col_action[key] is 'y'][0]
        for col in self.col_action:
            check_nan_percentage(self.df, col)
            if self.col_action[col] in ['ohe', 'se']:
                self.df = self.df.fillna(value={col: 'nan'})
        self.df = self.df.dropna(axis=0, subset=[c for c in self.xcols
                                                 if self.col_action[c]
                                                 is not 'del'] + [self.ycol])

        if n_rows == -1:
            self.df = self.df.sample(
                frac=1, random_state=sample_seed).reset_index(drop=True)
        else:
            self.df = self.df.sample(
                n=n_rows, random_state=sample_seed).reset_index(drop=True)
        if str_preprocess:
            self.df = preprocess_data(
                self.df, [key for key in self.col_action
                          if self.col_action[key] == 'se'])
        return

    def make_configs(self, **kw):
        if self.df is None:
            raise ValueError('need data to make column config')
        self.configs = [Config(name=name, kind=self.col_action.get(name), **kw)
                        for name in self.df.columns
                        if name in self.col_action.keys()]
        self.configs = [c for c in self.configs
                        if not (c.kind in ('del', 'y'))]

    def get_df(self, preprocess_df=True):
        if self.name == 'adult':
            header = ['age', 'workclass', 'fnlwgt', 'education',
                      'education-num', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'capital-gain',
                      'capital-loss', 'hours-per-week', 'native-country',
                      'income']
            df = pd.read_csv(self.file, names=header)
            df = df[df['occupation'] != ' ?']
            df = df.reset_index()
            df['income'] = (df['income'] == ' >50K')
            self.df = df
            self.col_action = {
                'age': 'num',
                'workclass': 'ohe',
                'fnlwgt': 'del',
                'education': 'ohe',
                'education-num': 'num',
                'marital-status': 'ohe',
                'occupation': 'se',
                'relationship': 'ohe',
                'race': 'ohe',
                'sex': 'ohe',
                'capital-gain': 'num',
                'capital-loss': 'num',
                'hours-per-week': 'num',
                'native-country': 'ohe',
                'income': 'y'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'binary-clf'

        if self.name == 'beer_reviews':
            df = pd.read_csv(self.file)
            df.shape
            self.df = df.dropna(axis=0, how='any')

            # print_unique_values(df)
            self.col_action = {
                'brewery_id': 'del',
                'brewery_name': 'del',
                'review_time': 'del',
                'review_overall': 'del',
                'review_aroma': 'num',
                'review_appearance': 'num',
                'review_profilename': 'del',
                'beer_style': 'y',
                'review_palate': 'num',
                'review_taste': 'num',
                'beer_name': 'se',
                'beer_abv': 'del',
                'beer_beerid': 'del'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'multiclass-clf'

        if self.name == 'midwest_survey':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'RespondentID': 'del',
                'In your own words, what would you call the part ' +
                'of the country you live in now?': 'se',
                'Personally identification as a Midwesterner?': 'ohe',
                'Illinois in MW?': 'ohe-1',
                'Indiana in MW?': 'ohe-1',
                'Iowa in MW?': 'ohe-1',
                'Kansas in MW?': 'ohe-1',
                'Michigan in MW?': 'ohe-1',
                'Minnesota in MW?': 'ohe-1',
                'Missouri in MW?': 'ohe-1',
                'Nebraska in MW?': 'ohe-1',
                'North Dakota in MW?': 'ohe-1',
                'Ohio in MW?': 'ohe-1',
                'South Dakota in MW?': 'ohe-1',
                'Wisconsin in MW?': 'ohe-1',
                'Arkansas in MW?': 'ohe-1',
                'Colorado in MW?': 'ohe-1',
                'Kentucky in MW?': 'ohe-1',
                'Oklahoma in MW?': 'ohe-1',
                'Pennsylvania in MW?': 'ohe-1',
                'West Virginia in MW?': 'ohe-1',
                'Montana in MW?': 'ohe-1',
                'Wyoming in MW?': 'ohe-1',
                'ZIP Code': 'del',
                'Gender': 'ohe',
                'Age': 'ohe',
                'Household Income': 'ohe',
                'Education': 'ohe',
                'Location (Census Region)': 'y'}
            le = LabelEncoder()
            ycol = [col for col in self.col_action
                    if self.col_action[col] == 'y']
            self.df[ycol] = le.fit_transform(self.df[ycol[0]].astype(str))
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'multiclass-clf'

        if self.name == 'indultos_espana':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'Fecha BOE': 'del',
                'Ministerio': 'ohe-1',
                'Ministro': 'ohe',
                'Partido en el Gobierno': 'ohe-1',
                'Género': 'ohe-1',
                'Tribunal': 'ohe',
                'Región': 'ohe',
                'Fecha Condena': 'del',
                'Rol en el delito': 'se',
                'Delito': 'se',
                'Año Inicio Delito': 'num',
                'Año Fin Delito': 'num',
                'Tipo de Indulto': 'y',
                'Fecha Indulto': 'del',
                'Categoría Cod.Penal': 'se',
                'Subcategoría Cod.Penal': 'se',
                'Fecha BOE.año': 'num',
                'Fecha BOE.mes': 'num',
                'Fecha BOE.día del mes': 'num',
                'Fecha BOE.día de la semana': 'num',
                'Fecha Condena.año': 'num',
                'Fecha Condena.mes': 'num',
                'Fecha Condena.día del mes': 'num',
                'Fecha Condena.día de la semana': 'num',
                'Fecha Indulto.año': 'num',
                'Fecha Indulto.mes': 'num',
                'Fecha Indulto.día del mes': 'num',
                'Fecha Indulto.día de la semana': 'num'}
            self.df['Tipo de Indulto'] = (
                self.df['Tipo de Indulto'] == 'indultar')
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'binary-clf'

        if self.name == 'docs_payments':
            # Variable names in Dollars for Docs dataset ######################
            pi_specialty = ['Physician_Specialty']
            drug_nm = ['Name_of_Associated_Covered_Drug_or_Biological1']
            #    'Name_of_Associated_Covered_Drug_or_Biological2',
            #    'Name_of_Associated_Covered_Drug_or_Biological3',
            #    'Name_of_Associated_Covered_Drug_or_Biological4',
            #    'Name_of_Associated_Covered_Drug_or_Biological5']
            dev_nm = ['Name_of_Associated_Covered_Device_or_Medical_Supply1']
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply2',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply3',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply4',
            #  'Name_of_Associated_Covered_Device_or_Medical_Supply5']
            corp = ['Applicable_Manufacturer_or_Applicable_GPO_Making_' +
                    'Payment_Name']
            amount = ['Total_Amount_of_Payment_USDollars']
            dispute = ['Dispute_Status_for_Publication']
            ###################################################################

            if os.path.exists(self.file):
                df = pd.read_hdf(self.file)
                # print('Loading DataFrame from:\n\t%s' % self.file)
            else:
                hdf_files = glob.glob(os.path.join(self.path, 'hdf', '*.h5'))
                hdf_files_ = []
                for file_ in hdf_files:
                    if 'RSRCH_PGYR2013' in file_:
                        hdf_files_.append(file_)
                    if 'GNRL_PGYR2013' in file_:
                        hdf_files_.append(file_)

                dfd_cols = pi_specialty + drug_nm + dev_nm + corp + amount + \
                    dispute
                df_dfd = pd.DataFrame(columns=dfd_cols)
                for hdf_file in hdf_files_:
                    if 'RSRCH' in hdf_file:
                        with pd.HDFStore(hdf_file) as hdf:
                            for key in hdf.keys():
                                df = pd.read_hdf(hdf_file, key)
                                df = df[dfd_cols]
                                df['status'] = 'allowed'
                                df = df.drop_duplicates(keep='first')
                                df_dfd = pd.concat([df_dfd, df],
                                                   ignore_index=True)
                                print('size: %d, %d' % tuple(df_dfd.shape))
                unique_vals = {}
                for col in df_dfd.columns:
                    unique_vals[col] = set(list(df_dfd[col].unique()))

                for hdf_file in hdf_files_:
                    if 'GNRL' in hdf_file:
                        with pd.HDFStore(hdf_file) as hdf:
                            for key in hdf.keys():
                                df = pd.read_hdf(hdf_file, key)
                                df = df[dfd_cols]
                                df['status'] = 'disallowed'
                                df = df.drop_duplicates(keep='first')
                                # remove all value thats are not in RSRCH
                                # for col in pi_specialty+drug_nm+dev_nm+corp:
                                #     print(col)
                                #     s1 = set(list(df[col].unique()))
                                #     s2 = unique_vals[col]
                                #     df = df.set_index(col).drop(labels=s1-s2)
                                #            .reset_index()
                                df_dfd = pd.concat([df_dfd, df],
                                                   ignore_index=True)
                                print('size: %d, %d' % tuple(df_dfd.shape))
                df_dfd = df_dfd.drop_duplicates(keep='first')
                df_dfd.to_hdf(self.file, 't1')
                df = df_dfd
            df['status'] = (df['status'] == 'allowed')
            self.df = df
            # print_unique_values(df)
            self.col_action = {
                pi_specialty[0]: 'del',
                drug_nm[0]: 'del',
                dev_nm[0]: 'del',
                corp[0]: 'se',
                amount[0]: 'num',
                dispute[0]: 'ohe-1',
                'status': 'y'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'binary-clf'

        if self.name == 'medical_charge':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'State': 'ohe',
                'Total population': 'del',
                'Median age': 'del',
                '% BachelorsDeg or higher': 'del',
                'Unemployment rate': 'del',
                'Per capita income': 'del',
                'Total households': 'del',
                'Average household size': 'del',
                '% Owner occupied housing': 'del',
                '% Renter occupied housing': 'del',
                '% Vacant housing': 'del',
                'Median home value': 'del',
                'Population growth 2010 to 2015 annual': 'del',
                'House hold growth 2010 to 2015 annual': 'del',
                'Per capita income growth 2010 to 2015 annual': 'del',
                '2012 state winner': 'del',
                'Medical procedure': 'se',
                'Total Discharges': 'del',
                'Average Covered Charges': 'num',
                'Average Total Payments': 'y'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'road_safety':
            files = self.file
            for filename in files:
                if filename.split('/')[-1] == '2015_Make_Model.csv':
                    df_mod = pd.read_csv(filename)
                    df_mod['Vehicle_Reference'] = (df_mod['Vehicle_Reference']
                                                   .map(str))
                    df_mod['Vehicle_Index'] = (df_mod['Accident_Index'] +
                                               df_mod['Vehicle_Reference'])
                    df_mod = df_mod.set_index('Vehicle_Index')
                    df_mod = df_mod.dropna(axis=0, how='any', subset=['make'])
            # for filename in files:
            #     if filename.split('/')[-1] == 'Accidents_2015.csv':
            #        df_acc = pd.read_csv(filename).set_index('Accident_Index')
            for filename in files:
                if filename.split('/')[-1] == 'Vehicles_2015.csv':
                    df_veh = pd.read_csv(filename)
                    df_veh['Vehicle_Reference'] = (df_veh['Vehicle_Reference']
                                                   .map(str))
                    df_veh['Vehicle_Index'] = (df_veh['Accident_Index'] +
                                               df_veh['Vehicle_Reference'])
                    df_veh = df_veh.set_index('Vehicle_Index')
            for filename in files:
                if filename.split('/')[-1] == 'Casualties_2015.csv':
                    df_cas = pd.read_csv(filename)
                    df_cas['Vehicle_Reference'] = (df_cas['Vehicle_Reference']
                                                   .map(str))
                    df_cas['Vehicle_Index'] = (df_cas['Accident_Index'] +
                                               df_cas['Vehicle_Reference'])
                    df_cas = df_cas.set_index('Vehicle_Index')

            df = df_cas.join(df_mod, how='left', lsuffix='_cas',
                             rsuffix='_model')
            df = df.dropna(axis=0, how='any', subset=['make'])
            df = df[df['Sex_of_Driver'] != 3]
            df = df[df['Sex_of_Driver'] != -1]
            df['Sex_of_Driver'] = df['Sex_of_Driver'] - 1
            self.df = df
            # print_unique_values(df)
            # col_action = {'Casualty_Severity': 'y',
            #               'Casualty_Class': 'num',
            #               'make': 'ohe',
            #               'model': 'se'}
            self.col_action = {
                'Sex_of_Driver': 'y',
                'model': 'se',
                'make': 'ohe'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'binary-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'
            self.file = self.file[0]

        if self.name == 'consumer_complaints':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'Date received': 'del',
                'Product': 'ohe',
                'Sub-product': 'ohe',
                'Issue': 'ohe',
                'Sub-issue': 'ohe',
                'Consumer complaint narrative': 'se',  # too long
                'Company public response': 'ohe',
                'Company': 'se',
                'State': 'del',
                'ZIP code': 'del',
                'Tags': 'del',
                'Consumer consent provided?': 'del',
                'Submitted via': 'ohe',
                'Date sent to company': 'del',
                'Company response to consumer': 'ohe',
                'Timely response?': 'ohe-1',
                'Consumer disputed?': 'y',
                'Complaint ID': 'del'
                          }
            if preprocess_df:
                self.preprocess()
            self.df = self.df.dropna(
                axis=0, how='any', subset=['Consumer disputed?'])
            self.df.loc[:, 'Consumer disputed?'] = (
                self.df['Consumer disputed?'] == 'Yes')
            self.clf_type = 'binary-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'traffic_violations':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'Date Of Stop': 'del',
                'Time Of Stop': 'del',
                'Agency': 'del',
                'SubAgency': 'del',  # 'ohe'
                'Description': 'se',
                'Location': 'del',
                'Latitude': 'del',
                'Longitude': 'del',
                'Accident': 'del',
                'Belts': 'ohe-1',
                'Personal Injury': 'del',
                'Property Damage': 'ohe-1',
                'Fatal': 'ohe-1',
                'Commercial License': 'ohe-1',
                'HAZMAT': 'ohe',
                'Commercial Vehicle': 'ohe-1',
                'Alcohol': 'ohe-1',
                'Work Zone': 'ohe-1',
                'State': 'del',  #
                'VehicleType': 'del',  # 'ohe'
                'Year': 'num',
                'Make': 'del',
                'Model': 'del',
                'Color': 'del',
                'Violation Type': 'y',
                'Charge': 'del',  # 'y'
                'Article': 'del',  # 'y'
                'Contributed To Accident': 'del',  # 'y'
                'Race': 'ohe',
                'Gender': 'ohe',
                'Driver City': 'del',
                'Driver State': 'del',
                'DL State': 'del',
                'Arrest Type': 'ohe',
                'Geolocation': 'del'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'crime_data':
            self.df = pd.read_csv(self.file)
            # print_unique_values(df)
            self.col_action = {
                'DR Number': 'del',
                'Date Reported': 'del',
                'Date Occurred': 'del',
                'Time Occurred': 'del',
                'Area ID': 'del',
                'Area Name': 'del',
                'Reporting District': 'del',
                'Crime Code': 'del',
                'Crime Code Description': 'y',
                'MO Codes': 'del',  # 'se'
                'Victim Age': 'num',
                'Victim Sex': 'ohe',
                'Victim Descent': 'ohe',
                'Premise Code': 'del',
                'Premise Description': 'ohe',
                'Weapon Used Code': 'del',
                'Weapon Description': 'ohe',
                'Status Code': 'del',
                'Status Description': 'del',
                'Crime Code 1': 'del',
                'Crime Code 2': 'del',
                'Crime Code 3': 'del',
                'Crime Code 4': 'del',
                'Address': 'del',
                'Cross Street': 'se',  # 'se'
                'Location ': 'del'
                          }
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'employee_salaries':
            df = pd.read_csv(self.file)
            self.col_action = {
                'Full Name': 'del',
                'Gender': 'ohe',
                'Current Annual Salary': 'y',
                '2016 Gross Pay Received': 'del',
                '2016 Overtime Pay': 'del',
                'Department': 'del',
                'Department Name': 'ohe',
                'Division': 'ohe',  # 'se'
                'Assignment Category': 'ohe-1',
                'Employee Position Title': 'se',
                'Underfilled Job Title': 'del',
                'Year First Hired': 'num'
                          }
            #Original#df['Current Annual Salary'] = [float(s[1:]) for s
            df['Current Annual Salary'] = [float(s) for s
                                           in df['Current Annual Salary']]
            df['Year First Hired'] = [datetime.datetime.strptime(
                str(d), '%m/%d/%Y').year for d
                                      in df['Date First Hired']]
            self.df = df
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'product_relevance':
            self.df = pd.read_csv(self.file, encoding='latin1')
            self.col_action = {
                'id': 'del',
                'product_uid': 'del',
                'product_title': 'se',
                'search_term': 'se',
                'relevance': 'y'}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        if self.name == 'federal_election':
            df_dict = pd.read_csv(self.data_dict_file)
            self.df = pd.read_csv(self.file, sep='|', encoding='latin1',
                                  header=None, names=df_dict['Column Name'])
            self.col_action = {
                'CMTE_ID': 'del',
                'AMNDT_IND': 'del',
                'RPT_TP': 'del',
                'TRANSACTION_PGI': 'ohe',
                'IMAGE_NUM': 'del',
                'TRANSACTION_TP': 'ohe',
                'ENTITY_TP': 'ohe',
                'NAME': 'del',
                'CITY': 'del',
                'STATE': 'ohe',
                'ZIP_CODE': 'del',
                'EMPLOYER': 'del',
                'OCCUPATION': 'se',  # 'se'
                'TRANSACTION_DT': 'del',
                'TRANSACTION_AMT': 'y',
                'OTHER_ID': 'del',
                'TRAN_ID': 'del',
                'FILE_NUM': 'del',
                'MEMO_CD': 'del',
                'MEMO_TEXT': 'se',  # 'se',
                'SUB_ID': 'del'}
            if preprocess_df:
                self.preprocess()
            # Some donations are negative
            self.df['TRANSACTION_AMT'] = self.df['TRANSACTION_AMT'].abs()
            # Predicting the log of the donation
            self.df['TRANSACTION_AMT'] = self.df[
                'TRANSACTION_AMT'].apply(np.log)
            self.df = self.df[self.df['TRANSACTION_AMT'] > 0]

            self.clf_type = 'regression'

        if self.name == 'public_procurement':
            self.df = pd.read_csv(self.file)
            self.col_action = {
                'ID_NOTICE_CAN': 'del',
                'YEAR': 'del',
                'ID_TYPE': 'ohe',  # (3/100000)
                'DT_DISPATCH': 'del',
                'XSD_VERSION': 'del',
                'CANCELLED': 'del',  # very unbalanced
                'CORRECTIONS': 'del',  # very unbalanced
                'B_MULTIPLE_CAE': 'del',
                'CAE_NAME': 'se',  # se (17388/100000)
                'CAE_NATIONALID': 'del',
                'CAE_ADDRESS': 'del',
                'CAE_TOWN': 'ohe',  # se (6184/100000)
                'CAE_POSTAL_CODE': 'del',
                'ISO_COUNTRY_CODE': 'ohe',
                'B_MULTIPLE_COUNTRY': 'del',  # (32/100000)
                'ISO_COUNTRY_CODE_ALL': 'del',
                'CAE_TYPE': 'ohe',  # (10/100000)
                'EU_INST_CODE': 'ohe',  # (11/100000)
                'MAIN_ACTIVITY': 'ohe',  # (200/100000)
                'B_ON_BEHALF': 'ohe',  # (3/100000)
                'B_INVOLVES_JOINT_PROCUREMENT': 'del',
                'B_AWARDED_BY_CENTRAL_BODY': 'del',
                'TYPE_OF_CONTRACT': 'ohe',  # (3/100000)
                'TAL_LOCATION_NUTS': 'del',  # (4238/542597)
                'B_FRA_AGREEMENT': 'ohe',  # (3/100000)
                'FRA_ESTIMATED': 'del',
                'B_FRA_CONTRACT': 'del',
                'B_DYN_PURCH_SYST': 'del',
                'CPV': 'ohe',  # 'ohe',  # (4325/542597)
                'ID_LOT': 'del',
                'ADDITIONAL_CPVS': 'del',
                'B_GPA': 'ohe',  # 'ohe'
                'LOTS_NUMBER': 'num',  # 'num',
                'VALUE_EURO': 'del',  # maybe this should be 'y'
                'VALUE_EURO_FIN_1': 'del',
                'VALUE_EURO_FIN_2': 'del',  # 'y'
                'B_EU_FUNDS': 'ohe',  # (3/100000)
                'TOP_TYPE': 'ohe',  # (9/100000)
                'B_ACCELERATED': 'del',
                'OUT_OF_DIRECTIVES': 'del',
                'CRIT_CODE': 'ohe',  # (3/100000)
                'CRIT_PRICE_WEIGHT': 'del',
                'CRIT_CRITERIA': 'del',  # 'not enough data'
                'CRIT_WEIGHTS': 'del',  # needs to be treated
                'B_ELECTRONIC_AUCTION': 'del',  # 'ohe',
                'NUMBER_AWARDS': 'num',
                'ID_AWARD': 'del',
                'ID_LOT_AWARDED': 'del',
                'INFO_ON_NON_AWARD': 'del',
                'INFO_UNPUBLISHED': 'del',
                'B_AWARDED_TO_A_GROUP': 'del',
                'WIN_NAME': 'del',  # (216535/542597)
                'WIN_NATIONALID': 'del',
                'WIN_ADDRESS': 'del',
                'WIN_TOWN': 'del',  # (38550/542597)
                'WIN_POSTAL_CODE': 'del',
                'WIN_COUNTRY_CODE': 'ohe',  # (81/100000)
                'B_CONTRACTOR_SME': 'del',
                'CONTRACT_NUMBER': 'del',
                'TITLE': 'del',  # 'se' (230882/542597)
                'NUMBER_OFFERS': 'del',  # num'; noisy
                'NUMBER_TENDERS_SME': 'del',
                'NUMBER_TENDERS_OTHER_EU': 'del',
                'NUMBER_TENDERS_NON_EU': 'del',
                'NUMBER_OFFERS_ELECTR': 'del',
                'AWARD_EST_VALUE_EURO': 'del',  # 'y'
                'AWARD_VALUE_EURO': 'y',
                'AWARD_VALUE_EURO_FIN_1': 'del',  # 'y'
                'B_SUBCONTRACTED': 'del',  # 'ohe'
                'DT_AWARD': 'del'}  # 'ohe'
            if preprocess_df:
                self.preprocess()
            ycol = [col for col in self.col_action
                    if self.col_action[col] == 'y'][0]
            self.df[ycol] = self.df[ycol].abs()
            # Predicting the log of the donation
            self.df[ycol] = self.df[ycol].apply(np.log)
            self.df = self.df[self.df[ycol] > 0]
            self.clf_type = 'regression'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'

        # add here info about the dataset #####################################
        if self.name == 'new_dataset':
            self.df = pd.read_csv(self.file)
            self.col_action = {}
            if preprocess_df:
                self.preprocess()
            self.clf_type = 'multiclass-clf'  # opts: 'regression',
            # 'binary-clf', 'multiclass-clf'
        #######################################################################
        self.df = self.df[list(self.col_action)]
        # why not but not coherent with the rest --> self.preprocess
        return self
