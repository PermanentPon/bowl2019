import pandas as pd
import os
import time
import pickle


class InputDataLoader:
    def __init__(self, data_folder, input_data_folder):
        self.data_folder = data_folder
        self.input_data_folder = input_data_folder
        self._train_df = None
        self._test_df = None
        self._train_labels_df = None
        self._specs_df = None
        self._sample_submission_df = None

    @property
    def train_df(self):
        if self._train_df is None:
            start_time = time.time()
            if 'train.pkl' in os.listdir(self.data_folder):
                self._train_df = pickle.load(open(os.path.join(self.data_folder, 'train.pkl'), 'rb'))
            else:
                self._train_df = pd.read_csv(os.path.join(self.input_data_folder, 'train.csv'), parse_dates=['timestamp'])
                pickle.dump(self._train_df, open(os.path.join(self.data_folder, 'train.pkl'), 'wb'))
                print(f'Saved training data to {os.path.join(self.data_folder, "train.pkl")} for future fast loading')
            print(f'Loaded train data for {round(time.time() - start_time, 2)} seconds.')
        return self._train_df

    @property
    def test_df(self):
        if self._test_df is None:
            start_time = time.time()
            if 'test.pkl' in os.listdir(self.data_folder):
                self._test_df = pickle.load(open(os.path.join(self.data_folder, 'test.pkl'), 'rb'))
            else:
                self._test_df = pd.read_csv(os.path.join(self.input_data_folder, 'test.csv'), parse_dates=['timestamp'])
                pickle.dump(self._test_df, open(os.path.join(self.data_folder, 'test.pkl'), 'wb'))
                print(f'Saved test_data data to {os.path.join(self.data_folder, "test.pkl")} for future fast loading')

            print(f'Loaded test data for {round(time.time() - start_time, 2)} seconds.')
        return self._test_df

    @property
    def train_labels_df(self):
        if self._train_labels_df is None:
            self._train_labels_df = pd.read_csv(os.path.join(self.input_data_folder, 'train_labels.csv'), index_col=0)
        return self._train_labels_df

    @property
    def specs_df(self):
        if self._specs_df is None:
            self._specs_df = pd.read_csv(os.path.join(self.input_data_folder, 'specs.csv'), index_col=0)
        return self._specs_df

    @property
    def sample_submission_df(self):
        if self._sample_submission_df is None:
            self._sample_submission_df = pd.read_csv(os.path.join(self.input_data_folder, 'sample_submission.csv'), index_col=0)
        return self._sample_submission_df
