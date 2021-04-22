import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns

MODEL_DIR = './model'                                  
if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)  
filename = os.path.join(MODEL_DIR, 'tmp_checkpoint.h5')

train_data = np.loadtxt("./sample_data/walmart/train.csv", dtype=np.str, delimiter=",")[1:,:]
test_data = np.loadtxt("./sample_data/walmart/test.csv", dtype=np.str, delimiter=",")[1:,:]
features = np.loadtxt("./sample_data/walmart/features.csv", dtype=np.str, delimiter=",")[1:,:]
stores = np.loadtxt("./sample_data/walmart/stores.csv", dtype=np.str, delimiter=",")[1:,:]

train_data = pd.DataFrame(train_data, columns=['Store','Dept','Date','Weekly_Sales','IsHoliday'])
test_data = pd.DataFrame(test_data, columns=['Store','Dept','Date','IsHoliday'])
features = pd.DataFrame(features, columns=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','IsHoliday'])
stores = pd.DataFrame(stores, columns=['Store','Type','Size'])

# 데이터 머지, 이후 필요없는 Type 컬럼 제거
train_merge = train_data.merge(features, how='left', on=['Store','Date','IsHoliday'])
test_merge = test_data.merge(features, how='left', on=['Store','Date','IsHoliday'])
full_train_data = train_merge.merge(stores, how='left', on=['Store'])
full_test_data = test_merge.merge(stores, how='left', on=['Store'])
full_train_data = full_train_data.drop(["Type"], axis=1)
full_test_data = full_test_data.drop(["Type"], axis=1)

# train data 날짜에서 년, 월, 일, 주차 추출하고 날짜 드롭함
full_train_data['Date'] = pd.to_datetime(full_train_data['Date'])
full_train_data["Year"] = full_train_data['Date'].dt.year
full_train_data["Month"]= full_train_data['Date'].dt.month
full_train_data["Day"]= full_train_data['Date'].dt.day
full_train_data["Week"] = full_train_data['Date'].dt.isocalendar().week
full_train_data = full_train_data.drop(["Date"], axis=1)

# test data 날짜에서 년, 월, 일, 주차 추출하고 날짜 드롭함
full_test_data['Date'] = pd.to_datetime(full_test_data['Date'])
full_test_data["Year"] = full_test_data['Date'].dt.year
full_test_data["Month"]= full_test_data['Date'].dt.month
full_test_data["Day"]= full_test_data['Date'].dt.day
full_test_data["Week"] = full_test_data['Date'].dt.isocalendar().week
full_test_data = full_test_data.drop(["Date"], axis=1)

pd.set_option('display.max_columns', 20)

# 데이터프레임 넘파이로 변환
full_train_data = full_train_data.values
full_test_data = full_test_data.values

# null값 0으로 바꿔줌
for i in range(len(full_train_data)):
    for j in range(len(full_train_data[i])):
        if full_train_data[i,j] == 'NA': full_train_data[i,j] = 0
for i in range(len(full_test_data)):
    for j in range(len(full_test_data[i])):
        if full_test_data[i,j] == 'NA': full_test_data[i,j] = 0

# train 전체 순서 정렬
n1 = full_train_data[:,:2]
n2 = full_train_data[:,2].reshape(-1,1)
n3 = full_train_data[:,3:14]
n4 = full_train_data[:,-4:]
full_train_data_2 = np.concatenate((n1,n4), axis=1)
full_train_data_2 = np.concatenate((full_train_data_2,n3), axis=1)
full_train_data_2 = np.concatenate((full_train_data_2,n2), axis=1)  

# test 전체 순서 정렬
m1 = full_test_data[:,:2]
m2 = full_test_data[:,2:13]
m3 = full_test_data[:,-4:]
full_test_data_2 = np.concatenate((m1,m3), axis=1)
full_test_data_2 = np.concatenate((full_test_data_2,m2), axis=1)  

# 추출용 ID 배열
Id = full_test_data_2[:, :5]

'''
train_corr = full_train_data_2.astype(float)
# Store, Year, Day, IsHoliday, Temperature, Fuel_Size, CPI, Unemployment는 상관계수 음수라서 버림
train_corr = pd.DataFrame(train_corr, columns=['Store','Dept','Year','Month','Day','Week','IsHoliday','Temperature','Fuel_Size','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','CPI','Unemployment','Size','Weekly_Sales'])
f, ax = plt.subplots(figsize=(20, 15))
mask = np.triu(np.ones_like(train_corr.corr(), dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(train_corr.corr(), cmap = cmap, mask=mask, vmax=.3, cbar_kws={"shrink": .5}, annot = True)
plt.show()
'''

# train 필요 열만 추출
tr_dept = full_train_data_2[:,1].reshape(-1,1)
tr_month = full_train_data_2[:,3].reshape(-1,1)
tr_week = full_train_data_2[:,5].reshape(-1,1) 
tr_mark = full_train_data_2[:, 9:14] 
tr_size_sales = full_train_data_2[:, -2:] 
full_train_data_2 = np.concatenate((tr_dept,tr_month), axis=1) 
full_train_data_2 = np.concatenate((full_train_data_2,tr_week), axis=1)
full_train_data_2 = np.concatenate((full_train_data_2,tr_mark), axis=1)
full_train_data_2 = np.concatenate((full_train_data_2,tr_size_sales), axis=1)

# test 필요 열만 추출
te_dept = full_test_data_2[:,1].reshape(-1,1)
te_month = full_test_data_2[:,3].reshape(-1,1)
te_week = full_test_data_2[:,5].reshape(-1,1)
te_mark = full_test_data_2[:, 9:14]
te_size = full_test_data_2[:, -1].reshape(-1,1)
full_test_data_2 = np.concatenate((te_dept,te_month), axis=1)
full_test_data_2 = np.concatenate((full_test_data_2,te_week), axis=1)
full_test_data_2 = np.concatenate((full_test_data_2,te_mark), axis=1)
full_test_data_2 = np.concatenate((full_test_data_2,te_size), axis=1)

'''
train_corr = full_train_data_2.astype(float)
# Store, Year, Day, IsHoliday, Temperature, Fuel_Size, CPI, Unemployment는 상관계수 음수라서 버림
train_corr = pd.DataFrame(train_corr, columns=['Dept','Month','Week','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5','Size','Weekly_Sales'])
f, ax = plt.subplots(figsize=(20, 15))
mask = np.triu(np.ones_like(train_corr.corr(), dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(train_corr.corr(), cmap = cmap, mask=mask, vmax=.3, cbar_kws={"shrink": .5})
plt.show()
'''


x_train = full_train_data_2[:,:9]
scaler1 = MinMaxScaler()
train_x = scaler1.fit_transform(x_train)

y_train = full_train_data_2[:,-1].reshape(-1,1)
scaler2 = MinMaxScaler()
train_y = scaler2.fit_transform(y_train)

x_test = full_test_data_2
test_x = scaler1.fit_transform(x_test)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print('\n')

X_train, X_valid, Y_train, Y_valid = train_test_split(train_x, train_y, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_dim=9, activation='relu'))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='Adam')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = filename, monitor = 'val_loss', verbose = 1, save_best_only = True)
history = model.fit(X_train, Y_train, epochs=20, batch_size=48, validation_data=(X_valid, Y_valid), callbacks=[early_stop, checkpointer])

model.load_weights(filename)
rf_pred = model.predict(test_x).reshape(-1,1)
label_y = scaler2.inverse_transform(rf_pred)

for i in range(len(full_test_data_2)):
    if len(str(Id[i,3])) == 1: Id[i,3] = '0' + str(Id[i,3])
    if len(str(Id[i,4])) == 1: Id[i,4] = '0' + str(Id[i,4])
    Id[i,0] = str(Id[i,0]) + "_" + str(Id[i,1]) + "_" + str(Id[i,2]) + '-' + str(Id[i,3]) +  '-' + str(Id[i,4])

weekly_s = label_y.astype(float)
output_id = Id[:,0].reshape(-1,1)

result_array = np.concatenate((output_id, weekly_s), axis=1)
submission = pd.DataFrame(result_array, columns=['Id','Weekly_Sales'])
submission.to_csv("Weekly_Sales_Prediction.csv", index=False)
