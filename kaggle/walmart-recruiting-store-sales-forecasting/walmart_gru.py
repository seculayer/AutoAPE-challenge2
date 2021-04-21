import pandas as pd
import numpy as np
import tensorflow as tf
import os

MODEL_DIR = './model'                          #먼저, 모델이 저장될 폴더를 지정해보자
if not os.path.exists(MODEL_DIR):               #해당 폴더가 있는지 확인하고 없으면 폴더를 만듦
    os.mkdir(MODEL_DIR)
filename = os.path.join(MODEL_DIR, 'tmp_checkpoint.h5')

train_data = np.loadtxt("./train.csv", dtype=np.str, delimiter=",")
test_data = np.loadtxt("./test.csv", dtype=np.str, delimiter=",")
features = np.loadtxt("./features.csv", dtype=np.str, delimiter=",")
stores = np.loadtxt("./stores.csv", dtype=np.str, delimiter=",")
train_data = train_data[1:,:]
test_data = test_data[1:,:]
features = features[1:,:]
stores = stores[1:,:]

train_data = pd.DataFrame(train_data, columns=['Store','Dept','Date','Weekly_Sales','IsHoliday'])
test_data = pd.DataFrame(test_data, columns=['Store','Dept','Date','IsHoliday'])
features = pd.DataFrame(features, columns=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
                                           'CPI','Unemployment','IsHoliday'])
stores = pd.DataFrame(stores, columns=['Store','Type','Size'])

train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])
features['Date'] = pd.to_datetime(features['Date'])

train_merge = train_data.merge(features, how='left', on=['Store','Date','IsHoliday'])
test_merge = test_data.merge(features, how='left', on=['Store','Date','IsHoliday'])
full_train_data = train_merge.merge(stores, how='left', on=['Store'])
full_test_data = test_merge.merge(stores, how='left', on=['Store'])

if full_train_data['IsHoliday'] is True: full_train_data['IsHoliday'] = 0
else: full_train_data['IsHoliday'] = 1
if full_test_data['IsHoliday'] is True: full_test_data['IsHoliday'] = 0
else: full_test_data['IsHoliday'] = 1

full_train_data['Type'] = full_train_data['Type'].replace({"A":1,"B":2,"C":3})
full_test_data['Type'] = full_test_data['Type'].replace({"A":1,"B":2,"C":3})

full_train_data["Year"] = full_train_data['Date'].dt.year
full_train_data["Month"] = full_train_data['Date'].dt.month
full_train_data["Day"] = full_train_data['Date'].dt.day
full_train_data["Week"] = full_train_data['Date'].dt.isocalendar().week
full_train_data = full_train_data.drop(["Date"], axis=1)

full_test_data["Year"] = full_test_data['Date'].dt.year
full_test_data["Month"] = full_test_data['Date'].dt.month
full_test_data["Day"] = full_test_data['Date'].dt.day
full_test_data["Week"] = full_test_data['Date'].dt.isocalendar().week

df_full_train_data = full_train_data[['Store','Dept','Year','Month','Day','Week','IsHoliday','Temperature','Fuel_Price',
                                   'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
                                   'CPI','Unemployment','Size','Type','Weekly_Sales']]
ex_full_test_data = full_test_data[['Store','Dept','Date']]
df_full_test_data = full_test_data[['Store','Dept','Year','Month','Day','Week','IsHoliday','Temperature','Fuel_Price',
                                 'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
                                 'CPI','Unemployment','Size','Type']]

full_train_data = df_full_train_data.values
full_test_data = df_full_test_data.values

full_train_data = np.where(full_train_data == 'NA', 0, full_train_data)
full_test_data = np.where(full_test_data == 'NA', 0, full_test_data)

for i in range(len(full_train_data)):
    for j in range(len(full_train_data[i])):
        full_train_data[i,j] = float(full_train_data[i,j])/1000000.

train_scaling = full_train_data
train_scaling = train_scaling.astype("float")

train_scaled = train_scaling[:290000,:]
valid_scaled = train_scaling[290000:,:]

train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(train_scaled, train_scaled, length=48, batch_size=10)
valid_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(valid_scaled, valid_scaled, length=39, batch_size=10)

#RNN(LSTM)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.GRU(32, activation='tanh', dropout = 0.5, return_sequences=True, input_shape=(48,train_scaled.shape[1])))
model.add(tf.keras.layers.GRU(64, activation='tanh', dropout = 0.3, return_sequences = False))
model.add(tf.keras.layers.Dense(train_scaled.shape[1]))
model.summary()

# SGD, RMSProp, Adagrad, Adam  tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = filename, monitor = 'val_loss', verbose = 1, save_best_only = True)
history = model.fit(train_gen, epochs=5, validation_data=valid_gen, callbacks=[early_stop, checkpointer])
''', checkpointer'''

for i in range(len(full_test_data)):
    for j in range(len(full_test_data[i])):
        full_test_data[i,j] = float(full_test_data[i,j])/1000000.

test_scaling = full_test_data
test_scaling = test_scaling.astype("float")

print(test_scaling.dtype)
print(test_scaling[:5,:])
print(test_scaling.shape)

pred_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(test_scaling, test_scaling, length=1, batch_size=10)
#pred_gen = test_scaling.reshape((48,test_scaling.shape[1]))
model.load_weights(filename)
pred = model.predict(pred_gen)

print(pred[:20,:])
print(pred.shape)
print(pred.dtype)

for i in range(len(pred)):
    for j in range(len(pred[i])):
        pred[i,j] = float(pred[i,j])*1000000.

inverse_pred = pred[:,-1]
print(inverse_pred)
inverse_pred = inverse_pred.astype(str).reshape(-1,1)
print(inverse_pred)

Id = pd.DataFrame(ex_full_test_data[["Store", "Dept","Date"]])
Id["Id"] = Id["Store"].astype(str) + "_" + Id["Dept"].astype(str) + "_" + Id["Date"].astype(str)
Id = pd.DataFrame(Id["Id"])

ex_array = inverse_pred[-1,0].reshape(-1,1)
inverse_pred = np.concatenate((inverse_pred, ex_array), axis=0)
print(inverse_pred.shape)

Id_array = Id.values
result_array = np.concatenate((Id, inverse_pred), axis=1)
submission = pd.DataFrame(result_array, columns=['Id','Weekly_Sales'])
submission.to_csv("Weekly_Sales_Prediction.csv", index=False)

'''
n_features = X_train_sc.shape[1]
test_predictions = []

first_eval_batch = X_train_sc[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test_rnn)):

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
    current_pred = model.predict(current_batch)[0]

    # store prediction
    test_predictions.append(current_pred)

    # update batch to now include prediction and drop first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    clear_output()
    print(len(test_predictions))
'''