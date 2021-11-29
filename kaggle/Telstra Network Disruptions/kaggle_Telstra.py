import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np


train_df = pd.read_csv('../moon/data/Telstra/train.csv')
test_df = pd.read_csv('../moon/data/Telstra/test.csv')
sample_submission = pd.read_csv('../moon/data/Telstra/sample_submission.csv')
event_type = pd.read_csv('../moon/data/Telstra/event_type.csv')
log_feature = pd.read_csv('../moon/data/Telstra/log_feature.csv')
resource_type = pd.read_csv('../moon/data/Telstra/resource_type.csv')
severity_type = pd.read_csv('../moon/data/Telstra/severity_type.csv')

event_type_oh = pd.get_dummies(event_type, columns=['event_type']).groupby(['id'], as_index=False).sum()
#print(event_type_oh.head())
resource_type_oh = pd.get_dummies(resource_type, columns=['resource_type']).groupby(['id'], as_index=False).sum()
#print(resource_type_oh.head())
log_feature_oh = pd.get_dummies(log_feature, columns=['log_feature']).groupby(['id'], as_index=False).sum()
#print(log_feature_oh.head())
severity_type_oh = pd.get_dummies(severity_type, columns=['severity_type']).groupby(['id'], as_index=False).sum()
#print(severity_type_oh.head())

# train
train = train_df.merge(event_type_oh, on=['id'])
train = train.merge(resource_type_oh, on=['id'])
train = train.merge(log_feature_oh, on=['id'])
train = train.merge(severity_type_oh, on=['id'])


# test
test = test_df.merge(event_type_oh, on=['id'])
test = test.merge(resource_type_oh, on=['id'])
test = test.merge(log_feature_oh, on=['id'])
test = test.merge(severity_type_oh, on=['id'])

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

target = train_df['fault_severity']
print(np.unique(target))
print(target.value_counts())

train = train.drop('fault_severity', axis=1)
print(train.columns)

le = LabelEncoder()
train_df['location'] = le.fit_transform(train_df['location'])
test_df['location'] = le.fit_transform(test_df['location'])
target = train_df['fault_severity']
train_df = train_df.drop('fault_severity', axis=1)

train['volume'] = train['volume'].astype('float64')
test['volume'] = test['volume'].astype('float64')
train['volume'] = np.log1p(train['volume'])
test['volume'] = np.log1p(test['volume'])

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=2000)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

def m_rf() :
	rf = RandomForestClassifier(n_estimators= 25, min_samples_split=30, max_depth=30, max_features=6, max_leaf_nodes=40, min_samples_leaf=10)
	rf.fit(x_train, y_train)
	pred = rf.predict_proba(x_test)
	predict_test = rf.predict_proba(test)
	print(predict_test)
	return predict_test
m_rf()

def rndS() :
	rf = RandomForestClassifier()
	param = {'n_estimators' : [23, 24, 25, 26, 27, 28, 29, 30],
			 'min_samples_split' : [20, 25, 30, 35],
			 'max_depth' : [10, 20, 30, 40],
			 'min_samples_leaf': [10, 20, 30, 40]}
	gr = GridSearchCV(rf, param_grid=param, cv=5)
	gr.fit(x_train, y_train)
	#print(pd.DataFrame(gr.cv_results_))
	print(gr.best_params_) # 최적 파라미터
	print(gr.best_score_) # 최고 점수
rndS()

def m_mlp() :
	solver = 'adam'
	hidden_layer_sizes = (40, 40)
	alpha = 0.02
	random_state = 1
	max_iter = 400
	learning_rate = 'invscaling'
	activation = 'relu'
	learning_rate_init = 0.1

	model2 = MLPClassifier(solver = solver, hidden_layer_sizes = hidden_layer_sizes,
						   alpha = alpha,random_state = random_state,
						   max_iter = max_iter, learning_rate = learning_rate,
						   learning_rate_init = learning_rate_init,
						   activation = activation)
	history = model2.fit(x_train, y_train)

	print("훈련세트 정확도 : {:.2f}".format(model2.score(x_train, y_train)))
	print("테스트 세트 정확도 : {:.2f}".format(model2.score(x_test, y_test)))

	predict_test = model2.predict_proba(test)
	print(predict_test)
	return predict_test
m_mlp()

def rndS2() :
	mlp = MLPClassifier()
	param = {'hidden_layer_sizes' : [(40,40)],
			 'alpha' : [0.2],
			 'random_state' : [1],
			 'max_iter': [400,500,600,700],
			 'learning_rate' : ['invscaling'],
			 'activation' : ['relu'],
			 'solver' : ['adam']}
	gr = GridSearchCV(mlp, param_grid=param, cv=3)
	gr.fit(x_train, y_train)
	#print(pd.DataFrame(gr.cv_results_))
	print(gr.best_params_) # 최적 파라미터
	print(gr.best_score_)# 최고 점수

# 모델 3 : DNN
early_stopping = EarlyStopping(patience=10)
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Dense(input_shape=(456,), units=516, activation='relu'))
model2.add(tf.keras.layers.Dense(units=512, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.3))
model2.add(tf.keras.layers.Dense(units=256, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.3))
model2.add(tf.keras.layers.Dense(units=128, activation='relu'))
model2.add(tf.keras.layers.Dropout(0.3))
model2.add(tf.keras.layers.Dense(units=3, activation='softmax'))
model2.summary()
model2.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

hist = model2.fit(x_train, y_train, epochs=60, batch_size=50, validation_data=(x_test, y_test))
y_pred = model2.predict_proba(test)
print(y_pred)


def chart(hist) :
	fig, loss_ax = plt.subplots()
	acc_ax = loss_ax.twinx()

	loss_ax.plot(hist.history['loss'], 'y', label='train loss')
	loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
	acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
	acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

	loss_ax.set_xlabel('epoch')
	loss_ax.set_ylabel('loss')
	acc_ax.set_ylabel('accuracy')

	loss_ax.legend(loc='upper left')
	acc_ax.legend(loc='lower left')

	plt.show()
#chart(hist)

