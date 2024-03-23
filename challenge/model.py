import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from datetime import datetime
import xgboost as xgb
import pickle


class DelayModel:
    

    def __init__(self):
        # I decided this is a good place to initialize the model as well.
        # Model should be saved in this attribute.
        self._model = xgb.XGBClassifier() 
        self.target = 'delay'
        self.disk_name = 'trained_model.sav'
    

    def _get_period_day(self, date: str):
        """
        Helper function for building the period_day feature.
        A given date can get 3 possible string values:
        'mañana' -> between 5:00 and 11:59
        'tarde'-> between 12:00 and 18:59
        'noche'-> between 19:00 and 4:59
        
        Args:
            date (str): String containing the date coming from the
                        'Fecha-I' column.

        Returns:
            str: 1 of 3 possible strings; 'mañana', 'tarde' or 'noche'.
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()

        if(date_time >= morning_min and date_time <= morning_max):
            return 'mañana'
        elif(date_time >= afternoon_min and date_time <= afternoon_max):
            return 'tarde'
        elif((date_time >= evening_min and date_time <= evening_max) or
            (date_time >= night_min and date_time <= night_max)):
            return 'noche'
        
       
    def _is_high_season(self, date: str):
        """
        Helper function for building the high_season feature.
        A given date can get 2 possible Int values:
        1 -> If date between Dec-15 and Mar-3, or Jul-15 and Jul-31, 
             or Sep-11 and Sep-30
        0 -> Otherwise 
        
        Args:
            date (str): String containing the date coming from the
                        'Fecha-I' column.

        Returns:
            Int: Either 1 or 0, 1 for 'high-season' dates and 0 
                 for 'non high-season' dates.
        """
        date_year = int(date.split('-')[0])
        date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = date_year)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = date_year, hour=23, minute=59, second=59)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = date_year)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = date_year, hour=23, minute=59, second=59)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = date_year)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = date_year, hour=23, minute=59, second=59)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = date_year)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = date_year, hour=23, minute=59, second=59)

        if ((date >= range1_min and date <= range1_max) or 
            (date >= range2_min and date <= range2_max) or 
            (date >= range3_min and date <= range3_max) or
            (date >= range4_min and date <= range4_max)):
            return 1
        else:
            return 0
        
        
    def _get_min_diff(self, data: pd.DataFrame):
        """
        Helper function for building the min_diff feature. This specific
        Feature is used to build the target column 'delay'.
        
        Args:
            data (pd.DataFrame): DF containing at least the 'Fecha-O'
                                 and the 'Fecha-I' columns.

        Returns:
            float64: difference in minutes between Fecha-O and Fecha-I
        """
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
    
    def _get_scale(self, target: pd.DataFrame):
        """
        Helper function for calculating the scale for balancing based
        on the target proportions.
        
        Args:
            target (pd.DataFrame): target.
        
        Returns:
            float64: scale to be used for balancing by the Model.
        """
        
        n_y0 = len(target[target[self.target] == 0])
        n_y1 = len(target[target[self.target] == 1])
        return round(n_y0/n_y1, 2)


    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predicting.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # We need to apply 'dummy tranformation' to the categorical features.
        # We needed to add some extra logic in order to make sure we create
        # and return all the neccessary features. 
        # For more details about this please take a look at the challenge.md
        # file, 2nd bullet point under the 'model.py' section.
        
        # These are the 10 features we need to create regarless if the category
        # is present on the given data.
        needed_mes = ['MES_4','MES_7', 'MES_10', 'MES_11', 'MES_12']
        needed_opera = ['OPERA_Latin American Wings', 'OPERA_Grupo LATAM',
                        'OPERA_Sky Airline', 'OPERA_Copa Air']
        needed_tipovuelo = ['TIPOVUELO_I']
        
        top_features = needed_mes + needed_opera + needed_tipovuelo
        
        # Getting the actual dummy columns for the 3 categorical features 
        opera_dum = pd.get_dummies(data['OPERA'], prefix = 'OPERA')
        tipovuelo_dum = pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO')
        mes_dum = pd.get_dummies(data['MES'], prefix = 'MES')
        
        # Getting the missing categories/features
        missing_opera = [opera for opera in needed_opera
                         if opera not in opera_dum.columns]
        missing_tipovuelo = [vuelo for vuelo in needed_tipovuelo
                             if vuelo not in tipovuelo_dum.columns]
        missing_mes = [mes for mes in needed_mes
                       if mes not in mes_dum.columns]
        
        # Adding the missing features to the corresponding DFs, notice
        # that the missing categories/features are added as columns with
        # 0 values only.
        for opera in missing_opera:
            opera_dum[opera] = 0
        
        for vuelo in missing_tipovuelo:
            tipovuelo_dum[vuelo] = 0
        
        for mes in missing_mes:
            mes_dum[mes] = 0
        
        # concatenating all the dummy features together 
        features = pd.concat([opera_dum, tipovuelo_dum, mes_dum], axis = 1)
        #print(features.columns)
        
        # Selecting the subset of features to be return to the caller
        features = features[top_features]
        
        # Checking if the caller requested the target column
        if target_column:
            target_df = pd. DataFrame() 
            data["min_diff"] = data.apply(self._get_min_diff, axis = 1)
            threshold_in_minutes = 15 # threshold to be consideres a delay
            target_df[self.target] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            return (features, target_df)
        else: # user only requested the input features
            return features
        
        

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # The first step is the get the scale for balancing 
        scale = self._get_scale(target)
        
        # We now define the model's paramaters dictionary
        params_xgb = {"learning_rate":0.01, "scale_pos_weight":scale}
        
        # setting the parameters to the model's object.
        self._model = self._model.set_params(**params_xgb)
        
        # Training the actual model using the given features and 
        # target data.
        self._model = self._model.fit(features, target[self.target])
        
        # save the trained model to disk 
        pickle.dump(self._model, open(self.disk_name, 'wb'))
        
                

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
            
        """
        # Please take a look at the comment # 4 on the model.py section
        # of the challenge.md file. 
        
        # load the pre-trained model from disk
        loaded_model = pickle.load(open(self.disk_name, 'rb'))
        predictions = loaded_model.predict(features)
        
        # Until this point the predictions have an int32 dtype, however
        # the test cases are expecting int as datatype
        
        predictions = [int(pred) for pred in predictions]
        
        # This approach did not pass the test cases:
        # predictions = self._model.predict(features)
        
        return predictions
       