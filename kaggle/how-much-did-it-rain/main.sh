python standard_features.py

./make_binned_features_train.sh
./make_binned_features_test.sh

#train the classifiers
python ./train_subset1.py
python ./train_subset2.py
python ./train_subset3.py
python ./train_subset4.py
python ./train_subset5.py

#create the submission file. 
python make_predictions.py

