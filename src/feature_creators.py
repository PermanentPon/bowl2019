import time
import ast
import multiprocessing
import json

from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from category_encoders.ordinal import OrdinalEncoder
from joblib import Parallel, delayed

from utils import paralize_df


class UserFeaturizer:
    def __init__(self):
        self.titles_dict = dict()

    def calculate_assessments_features(self, user_sample):
        user_features = dict()
        titles = [title for title, type in self.titles_dict.items() if type in ['Game', 'Assessment']]
        for title in titles:
            user_features[f"{title.lower().replace(' ', '_')}_misses"] = None
            user_features[f"{title.lower().replace(' ', '_')}_duration"] = None


        # both assessments and games have tasks which children should accomplish and they assessed
        assessed_results = user_sample[(user_sample.event_code == 2030) & (user_sample.type.isin(['Game', 'Assessment']))]
        for title, results in assessed_results.groupby('title'):
            misses = [json.loads(event_data)['misses'] for event_data in results.event_data.tolist()]
            if results.iloc[0, :].type == 'Assessment':
                duration = sum(user_sample[(user_sample.event_code == 2010) & (
                               user_sample.title == title)].game_time.tolist())
            else:
                durations = results.loc[results.groupby('game_session').timestamp.idxmax(), 'game_time'].tolist()
                duration = sum(durations)

            user_features[f"{title.lower().replace(' ', '_')}_misses"] = sum(misses)
            user_features[f"{title.lower().replace(' ', '_')}_misses_duration"] = sum(misses)/(duration+1)
            user_features[f"{title.lower().replace(' ', '_')}_duration"] = duration
        return user_features

    def calculate_placement_accuracy_features(self, user_sample):
        user_features = dict()
        titles = [title for title, type in self.titles_dict.items() if type == 'Assessment']
        for title in titles:
            user_features[f"{title.lower().replace(' ', '_')}_placement_acc"] = None
        placement_results = user_sample[((user_sample.event_code == 4100) | (user_sample.event_code == 4020) | (user_sample.event_code == 4025)) & (user_sample.title.isin(titles))]
        for title, results in placement_results.groupby('title'):
            true_attempts = results['event_data'].str.contains(':true,').sum()
            false_attempts = results['event_data'].str.contains('false').sum()
            user_features[f"{title.lower().replace(' ', '_')}_placement_acc"] = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
        return user_features

    def calculate_basic_stat_features(self, user_sample):
        features = dict()
        for title, results in user_sample.groupby('title'):
            # print('All:', results.shape)
            features[f"{title}_count"] = results['game_session'].count()
            results = results.loc[results.groupby('game_session').timestamp.idxmax(), :]
            for column in ['event_count', 'game_time']:
                # print('last:', results.shape)
                features[f"{title}_sessions_count"] = results.shape[0]
                # features[f"{title}_{event_code}_{column}_max"] = results[column].max()
                # features[f"{title}_{event_code}_{column}_min"] = results[column].min()
                features[f"{title}_{column}_mean"] = results[column].mean()
        return features

    def calculate_basic_stat_features(self, user_sample):
        features = dict()
        for (title, event_code), results in user_sample.groupby(['title', 'event_code']):
            # print('All:', results.shape)
            features[f"{title}_{event_code}_count"] = results['game_session'].count()
            results = results.loc[results.groupby('game_session').timestamp.idxmax(), :]
            for column in ['event_count', 'game_time']:
                # print('last:', results.shape)
                features[f"{title}_{event_code}_sessions_count"] = results.shape[0]
                # features[f"{title}_{event_code}_{column}_max"] = results[column].max()
                # features[f"{title}_{event_code}_{column}_min"] = results[column].min()
                features[f"{title}_{event_code}_{column}_mean"] = results[column].mean()
        return features


    def calculate_placement_accuracy_feature_current_assessment(self, title, user_sample):
        user_features = dict()

        user_features["current_assessment__placement_acc"] = None
        placement_results = user_sample[
            ((user_sample.event_code == 4020) | (user_sample.event_code == 4025)) & (user_sample.title == 'title')]
        results = placement_results
        true_attempts = results['event_data'].str.contains('true').sum()
        false_attempts = results['event_data'].str.contains('false').sum()
        user_features["current_assessment__placement_acc"] = true_attempts / (
                    true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else -1
        return user_features

    def calculate_per_user_features(self, user_sample, game_session, accuracy_group):
        user_features = dict()
        assessment_start_event = user_sample[(user_sample.game_session == game_session) & (user_sample.event_code == 2000)].iloc[0,:]
        assessment_start_time = assessment_start_event.timestamp
        user_sample_before = user_sample[user_sample.timestamp < assessment_start_time]

        user_features.update(self.calculate_assessments_features(user_sample_before))
        user_features.update(self.calculate_placement_accuracy_features(user_sample_before))
        user_features.update(self.calculate_basic_stat_features(user_sample_before))

        user_features['assessment_title'] = assessment_start_event.title
        user_features['world'] = assessment_start_event.world
        return user_features, accuracy_group

    @staticmethod
    def yield_train_user_sample(train_df, train_labels_df):
        train_df_indexed = train_df.set_index('installation_id')
        for game_session, row in tqdm(train_labels_df.iterrows(), total=train_labels_df.shape[0]):
            user_sample = train_df_indexed.loc[row.installation_id, :]
            yield user_sample, game_session, row.accuracy_group

    @staticmethod
    def yield_test_user_sample(test_df):
        for installation_id, user_sample in test_df.groupby(['installation_id']):
            game_session = user_sample.iloc[-1, :].game_session
            yield user_sample, game_session, installation_id

    def calculate_all_train_user_features(self, train_df, train_labels_df, n_jobs=multiprocessing.cpu_count()):
        titles_list = train_df.title.unique()
        for title in titles_list:
            self.titles_dict[title] = train_df[train_df.title == title].iloc[0,:].type
        result = Parallel(n_jobs=n_jobs)(delayed(self.calculate_per_user_features)(user_sample, game_session, accuracy_group) for user_sample, game_session, accuracy_group in self.yield_train_user_sample(train_df, train_labels_df))
        user_features, accuracy_group = zip(*result)
        return user_features, accuracy_group

    def calculate_all_test_user_features(self, test_df, n_jobs=multiprocessing.cpu_count()):
        result = Parallel(n_jobs=n_jobs)(delayed(self.calculate_per_user_features)(user_sample, game_session, installation_id) for user_sample, game_session, installation_id in self.yield_test_user_sample(test_df))
        user_features, installation_ids = zip(*result)
        return user_features, installation_ids


# Following classes aren't used yet
class BasicFeaturizer(BaseEstimator, TransformerMixin):
    """
    Class to combine all the methods for preprocessing steps and creating basic features.
    Basic features are ones that are calculated the same way for train and test data.
    So you initialize the class once, don't need to fit, and run transform straight away on all your data in dataframes.
    """
    def __init__(self, features):
        self.feature_methods = []
        for feature in features:
            if feature + '_' not in dir(self):
                raise Exception(
                    f'You haven\'t implemented {feature}. Add a method to Dataset class which creates the feature')
            feature_method = getattr(self, feature + '_')
            self.feature_methods.append(feature_method)

    def fit(self):
        pass

    def transform(self, df: pd.DataFrame):
        print('Running preprocessing and basic features creation with BasicFeaturizer...')
        initial_columns = df.columns
        df_list = []
        start_time = time.time()
        for feature_method in self.feature_methods:
            print(f'Running feature creation for {feature_method.__name__} method')
            df = paralize_df(df, feature_method)

            new_columns = list(set(df.columns) - set(initial_columns))
            if not all([column.startswith(feature_method.__name__[:-1]) for column in new_columns]):
                raise Exception(f'All new features should start with a name of a method they were created in. '
                                f'Method {feature_method.__name__} does\'t comply with this rule! Created features - {new_columns}')

            print(f'{feature_method.__name__} method. {new_columns} features created')
            df_list.append(df)
        print(f'Preprocessing executed for {round(time.time() - start_time, 2)} seconds.')
        print('-' * 100, '\n')
        return pd.concat(df_list, axis=1)

    @staticmethod
    def time_features_(df):
        df = df.assign(
            time_features_hour=df.timestamp.dt.hour,
            time_features_day=df.timestamp.dt.day,
            time_features_month=df.timestamp.dt.month,
            time_features_year=df.timestamp.dt.year)
        return df


class TrainableEncoders(BaseEstimator, TransformerMixin):
    """
    Combines all encoders. Has to be fitted on all data.
    """
    def __init__(self):
        self.title_oe = OrdinalEncoder()
        self.world_oe = OrdinalEncoder()
        # Add more encoder objects here, also add fit ad transform code

    def fit(self, df: pd.DataFrame):
        self.title_oe.fit(list(set(df['title'].unique())))
        self.world_oe.fit(list(set(df['world'].unique())))

    def transform(self, df: pd.DataFrame):
        df['title'] = self.title_oe.transform(df['title'].values)
        df['world'] = self.world_oe.transform(df['world'].values)
        return df
