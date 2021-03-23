import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":

    X_train = pd.read_csv("../203_biol/data/train.csv")
    X_test = pd.read_csv("../203_biol/data/test.csv")

    y_train = X_train["Activity"]

    X_train.drop(columns=["Activity"], inplace=True)

    columns = X_train.columns

    # Normalizing column values
    for col in columns:
        MinMax = MinMaxScaler()

        X_train_arr = X_train[col].astype(float).values
        X_test_arr = X_test[col].astype(float).values

        X_train_arr = MinMax.fit_transform(X_train_arr.reshape(-1, 1))
        X_test_arr = MinMax.transform(X_test_arr.reshape(-1, 1))

        X_train[col] = X_train_arr
        X_test[col] = X_test_arr

    model = LGBMClassifier(num_trees=5000, #early_stopping_round=300,
                           random_state=314,
                           learning_rate=0.005, num_threads=8, silent=True, n_jobs=2,
                           subsample_freq=20, feature_fraction=0.9, bagging_fraction=0.9,
                           min_data_in_leaf=3, min_sum_hessian_in_leaf=30.0, is_enable_sparse=True,
                           use_two_round_loading=False, subsample=0.9, max_bin=10, metric_freq=1)

    model.fit(X_train, y_train)
    predicted_prob = model.predict_proba(X_test)
    Probability = predicted_prob[:, 1]
    MoleculeId = np.array(range(1, len(X_test) + 1))
    submission = pd.DataFrame()
    submission["MoleculeId"] = MoleculeId
    submission['PredictedProbability'] = Probability
    submission.to_csv('submission.csv', index=None)