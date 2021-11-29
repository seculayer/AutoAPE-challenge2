import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


df= pd.read_csv('../moon/data/Otto/train.csv')
test = pd.read_csv('../moon/data/Otto/test.csv')
columns = df.columns[1:-1]
train = df[columns]
target = df['target']
target = LabelEncoder().fit_transform(target)
test = test.drop('id', axis=1)

train['f_sum'] = train[columns].sum(axis=1)
train['f_non_zero'] = (train[columns] > 0).sum(axis=1)
test = test.drop('id', axis=1)
test['f_sum'] = test.values.sum(axis=1)
test['f_non_zero'] = (test.values > 0).sum(axis=1)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for train_index, test_index in sss.split(train.values, target) :
	x_train = train.values[train_index]
	x_test = train.values[test_index]
	y_train = target[train_index]
	y_test = target[test_index]

def model1(x_train, y_train, test) :
	n_estimators = 300
	max_features = 30
	max_depth = 100
	model1 = RandomForestClassifier(n_estimators=n_estimators,
									max_features=max_features,
									max_depth=max_depth, random_state=42)
	model1.fit(x_train, y_train)
	prediction = model1.predict_proba(test)

model2 = MLPClassifier(solver='sgd', hidden_layer_sizes=(40, 30), alpha= 0.02,
					   random_state=1000, max_iter=400, learning_rate='adaptive',
					   learning_rate_init=0.001, activation='relu')
model2.fit(x_train, y_train)

print("훈련세트 정확도 : {:.2f}".format(model2.score(x_train, y_train)))
print("테스트 세트 정확도 : {:.2f}".format(model2.score(x_test, y_test)))

test_prob = model2.predict_proba(test)

model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.Dense(input_shape=(93, ), units=512 ,activation='relu'))
model3.add(tf.keras.layers.Dropout(0.3))
model3.add(tf.keras.layers.Dense(units=256, activation='relu'))
model3.add(tf.keras.layers.Dropout(0.3))
model3.add(tf.keras.layers.Dense(units=512, activation='relu'))
model3.add(tf.keras.layers.Dropout(0.3))
model3.add(tf.keras.layers.Dense(units=256, activation='relu'))
model3.add(tf.keras.layers.Dropout(0.3))
model3.add(tf.keras.layers.Dense(units=9, activation='softmax'))
model3.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
model3.summary()

hist = model3.fit(x_train, y_train, epochs=15, batch_size=500, validation_data=(x_test, y_test))

y_pred = model3.predict(test)

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

chart(hist)

corr = df.corr()
print(corr)
plt.matshow(train.corr())
plt.colorbar()
plt.show()

# 주성분 분석
x= StandardScaler().fit_transform(train)
t = StandardScaler().fit_transform(test)
x = pd.DataFrame(data=x, columns=train.columns)
t = pd.DataFrame(data=t, columns=test.columns)

pca = PCA(n_components=2)
pc = pca.fit_transform(x)
pc_t = pca.fit_transform(t)
pcdf = pd.DataFrame(data = pc, columns=['pc1', 'pc2'])
pcdf_t = pd.DataFrame(data = pc_t, columns=['pc1', 'pc2'])
print(pcdf.shape)
print(pcdf_t.shape)
finalDf = pd.concat([pcdf, df[['target']]], axis=1)
print(finalDf.head())
print('주성분 설명력 : ', pca.explained_variance_ratio_)

# REF 적용  Random Forest
# RF용 StandardScaler
ss = StandardScaler()
x_train_ss = ss.fit_transform(x_train)
x_test_ss = ss.fit_transform(x_test)
test_ss = ss.fit_transform(test)
print(train.shape, test.shape)

#sub model 1 : RandomForest
forest = RandomForestClassifier(n_estimators= 500, random_state=7)
select = RFE(forest, n_features_to_select= 77)
x_train_rf = select.fit_transform(x_train_ss, y_train)
x_test_rf = select.transform(x_test_ss)
test_rf = select.transform(test_ss)
print(x_train_rf.shape)
score = select.fit(x_train_rf, y_train).score(x_test_rf, y_test)
print('RFE 후 acc : {:.3f}'.format(score))

rf_y_pred = select.predict_proba(test_rf)
