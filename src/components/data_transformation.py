import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        pass

    def time_to_minutes(self, t_series):
        # t_series contains HHMM as float or int, e.g. 441.0
        t_series = t_series.fillna(0).astype(int)
        return (t_series // 100) * 60 + (t_series % 100)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column_name = "class"

            def transform_df(df, is_train=True, hist_intensity=None):
                # Clean up classes
                df = df[df['class'].isin(['C', 'M', 'X'])].copy()
                
                # Parse times
                df['start_time_min'] = self.time_to_minutes(df['start_time'])
                df['end_time_min'] = self.time_to_minutes(df['end_time'])
                df['peak_time_min'] = self.time_to_minutes(df['peak_time'])

                # Durations
                df['flare_duration'] = df['end_time_min'] - df['start_time_min']
                df.loc[df['flare_duration'] < 0, 'flare_duration'] += 24 * 60

                df['time_to_peak'] = df['peak_time_min'] - df['start_time_min']
                df.loc[df['time_to_peak'] < 0, 'time_to_peak'] += 24 * 60

                # Start hour
                df['start_hour'] = df['start_time'].fillna(0).astype(int) // 100

                # Date parsing
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['start_month'] = df['date'].dt.month.fillna(0).astype(int)
                df['start_dayofweek'] = df['date'].dt.dayofweek.fillna(0).astype(int)

                # Region avg intensity
                df['region'] = df['region'].fillna('UNKNOWN')
                if is_train:
                    hist_intensity = df.groupby('region')['intensity'].mean().reset_index()
                    hist_intensity.rename(columns={'intensity': 'avg_intensity'}, inplace=True)
                
                df = pd.merge(df, hist_intensity, on='region', how='left')
                df['avg_intensity'] = df['avg_intensity'].fillna(df['intensity'].mean())

                # Label encoding
                le_region = LabelEncoder()
                df['region'] = le_region.fit_transform(df['region'])
                
                le_obs = LabelEncoder()
                df['observatory'] = df['observatory'].fillna('UNKNOWN')
                df['observatory'] = le_obs.fit_transform(df['observatory'])

                # Drop unnecessary columns
                cols_to_drop = ['start_time', 'end_time', 'peak_time', 'extra', 'flarenumber', 'date', 
                                'start_time_min', 'end_time_min', 'peak_time_min']
                df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
                
                return df, hist_intensity

            train_df, hist_intensity = transform_df(train_df, is_train=True)
            test_df, _ = transform_df(test_df, is_train=False, hist_intensity=hist_intensity)

            # Map target
            target_map = {'C': 0, 'M': 1, 'X': 2}
            train_df['class'] = train_df['class'].map(target_map)
            test_df['class'] = test_df['class'].map(target_map)
            
            # Drop rows with NaNs in target
            train_df = train_df.dropna(subset=['class'])
            test_df = test_df.dropna(subset=['class'])

            # Split X and y
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name]

            # SMOTE for class imbalance
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            train_arr = np.c_[X_train_resampled, np.array(y_train_resampled)]
            test_arr = np.c_[X_test, np.array(y_test)]

            return train_arr, test_arr, X_train.columns.tolist()

        except Exception as e:
            print(f"Error in data transformation: {e}")
            raise e
