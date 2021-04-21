import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")

test_data = pd.read_csv("test.csv")
test_data_id = test_data['id']

print(train_data.info())
print(train_data.head())

y = train_data['hand']
y = y.values
y = y.reshape(-1, 1)
y_encode = to_categorical(y)

def hearts(x):
    if x == 1:
        return 1
    else:
        return 0
def spades(x):
    if x == 2:
        return 1
    else:
        return 0
def clubs(x):
    if x == 3:
        return 1
    else:
        return 0
def diamonds(x):
    if x == 4:
        return 1
    else:
        return 0


train_data['hearts'] = 0 + train_data['S1'].apply(hearts) + train_data['S2'].apply(hearts) + train_data['S3'].apply(hearts) + train_data['S4'].apply(hearts) + train_data['S5'].apply(hearts)
train_data['spades'] = 0 + train_data['S1'].apply(spades) + train_data['S2'].apply(spades) + train_data['S3'].apply(spades) + train_data['S4'].apply(spades) + train_data['S5'].apply(spades)
train_data['clubs'] = 0 + train_data['S1'].apply(clubs) + train_data['S2'].apply(clubs) + train_data['S3'].apply(clubs) + train_data['S4'].apply(clubs) + train_data['S5'].apply(clubs)
train_data['diamonds'] = 0 + train_data['S1'].apply(diamonds) + train_data['S2'].apply(diamonds) + train_data['S3'].apply(diamonds) + train_data['S4'].apply(diamonds) + train_data['S5'].apply(diamonds)

sub = ['C1','C2','C3','C4','C5']
sub_sort = train_data[sub]

for i in range(sub_sort.shape[0]):
    temp = sub_sort.iloc[i]
    temp_list = list(temp)
    temp_list.sort()

    for x in range(len(temp)):
        sub_sort.iloc[i][x] = temp_list[x]


train_data['diff_1'] = sub_sort['C2'] - sub_sort['C1']
train_data['diff_2'] = sub_sort['C3'] - sub_sort['C2']
train_data['diff_3'] = sub_sort['C4'] - sub_sort['C3']
train_data['diff_4'] = sub_sort['C5'] - sub_sort['C4']
train_data['diff_5'] = sub_sort['C5'] - sub_sort['C1']


feature = ['hearts','spades','clubs','diamonds','diff_1','diff_2','diff_3','diff_4','diff_5']
X1 = train_data[feature]
X1 = X1.values
x_train, x_val, y_train, y_val = train_test_split(X1, y_encode, test_size=0.2, random_state=100)

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=x_train.shape[1:]))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

model.summary()

history = model.fit(x_train, y_train, batch_size=100, epochs=150,
                    validation_data=(x_val, y_val))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()

test_data['hearts'] = 0 + test_data['S1'].apply(hearts) + test_data['S2'].apply(hearts) + test_data['S3'].apply(hearts) + test_data['S4'].apply(hearts) + test_data['S5'].apply(hearts)
test_data['spades'] = 0 + test_data['S1'].apply(spades) + test_data['S2'].apply(spades) + test_data['S3'].apply(spades) + test_data['S4'].apply(spades) + test_data['S5'].apply(spades)
test_data['clubs'] = 0 + test_data['S1'].apply(clubs) + test_data['S2'].apply(clubs) + test_data['S3'].apply(clubs) + test_data['S4'].apply(clubs) + test_data['S5'].apply(clubs)
test_data['diamonds'] = 0 + test_data['S1'].apply(diamonds) + test_data['S2'].apply(diamonds) + test_data['S3'].apply(diamonds) + test_data['S4'].apply(diamonds) + test_data['S5'].apply(diamonds)

test_sort = test_data[sub]

## test.csv 카드번호 오름차순 정렬
for j in range(test_sort.shape[0]):
    temp = test_sort.iloc[j]
    temp_list = list(temp)
    temp_list.sort()

    for x in range(len(temp)):
        test_sort.iloc[j][x] = temp_list[x]

test_data['diff_1'] = test_sort['C2'] - test_sort['C1']
test_data['diff_2'] = test_sort['C3'] - test_sort['C2']
test_data['diff_3'] = test_sort['C4'] - test_sort['C3']
test_data['diff_4'] = test_sort['C5'] - test_sort['C4']
test_data['diff_5'] = test_sort['C5'] - test_sort['C1']

test_set = test_data[feature]
test_np = test_set.values

result = model.predict(test_np)
result = np.argmax(result, axis=1)

output = pd.DataFrame({'id': test_data_id, 'hand': result})
output.to_csv('./OH_encoding_submission_2.csv', index=False)
