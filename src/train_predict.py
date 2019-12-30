import click
import pandas as pd
import numpy as np

from data_loader import InputDataLoader
from feature_creators import UserFeaturizer
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score


def load_preprocess_data(data_folder, input_data_folder):
    input_data = InputDataLoader(data_folder, input_data_folder)

    # basic_featurizer = BasicFeaturizer(['time_features'])
    # train_df = basic_featurizer.transform(input_data.train_df)
    # test_df = basic_featurizer.transform(input_data.test_df)
    #
    # trainable_encoders = TrainableEncoders()
    # trainable_encoders.fit(pd.concat([train_df, test_df]))
    # train_df = trainable_encoders.transform(train_df)
    # test_df = trainable_encoders.transform(test_df)

    user_featurizer = UserFeaturizer()
    train_features, train_accuracy_group = user_featurizer.calculate_all_train_user_features(input_data.train_df, input_data.train_labels_df)
    test_features, test_installation_ids = user_featurizer.calculate_all_test_user_features(input_data.test_df)

    return train_features, train_accuracy_group, test_features, test_installation_ids


def get_catboost_model():
    model = CatBoostClassifier(
        iterations=10000,
        random_seed=111,
        loss_function='MultiClass',
        eval_metric='WKappa',
        # learning_rate=0.01,
        logging_level='Silent',
        task_type="CPU",
        devices='0',
        nan_mode='Max'
        # max_depth=5,
        # l2_leaf_reg=3.0,
        # bagging_temperature=1
    )
    return model

@click.command()
@click.option('--data_folder', default='data')
@click.option('--input_data_folder', default='data/input')
def main(data_folder, input_data_folder):
    train_features, train_accuracy_group, test_features, test_installation_ids = \
        load_preprocess_data(data_folder, input_data_folder)
    print(train_features[1])
    print(test_features[1])

    train_features_df = pd.DataFrame(train_features)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=111)
    for fold_i, (train_index, valid_index) in enumerate(skf.split(train_features_df, train_accuracy_group)):
        X_train, X_valid = train_features_df.iloc[train_index], train_features_df.iloc[valid_index]
        y_train, y_valid = np.array(train_accuracy_group)[train_index], np.array(train_accuracy_group)[valid_index]

        model = get_catboost_model()

        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=[train_features_df.columns.to_list().index('assessment_title'),
                          train_features_df.columns.to_list().index('world')]);
        print(f'Score: {model.get_best_score()}, iteration: {model.get_best_iteration()}')
        y_pred = np.round(model.predict(X_valid)).astype('int').squeeze()
        print(f'Final WKappa score: {cohen_kappa_score(y_pred, y_valid, weights="quadratic")}')

        break


if __name__=='__main__':
    main()