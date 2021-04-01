import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

X_train_df=pd.read_csv('X_train.csv')
y_train_df=pd.read_csv('y_train.csv')
X_test_df=pd.read_csv('X_test.csv')
y_train_df=y_train_df[['series_id','surface']]

#EDA
check_describe=X_train_df.describe()
plt.figure(figsize = (12, 5))
plt.plot(X_train_df['orientation_W'][:128])
plt.title('first sample orientationW')
plt.show()

#종속변수 개수 확인, 훈련할 때 집단별 샘플링 필요
y_train_df['surface'].value_counts()
plt.figure(figsize = (12, 5))
plt.title('surface')
y_train_df['surface'].value_counts().plot(kind='bar')
plt.show()

#독립 변수 사이 상관관계
need_col=X_train_df.columns.difference(['row_id','series_id','measurement_number'])
X_corr=X_train_df[need_col].corr()


#surface 별로 평균값,분산 보기
need_col=X_train_df.columns.difference(['row_id','measurement_number'])
full_data=pd.merge(X_train_df[need_col],y_train_df,how='inner',on='series_id')
check_mean=full_data.groupby(by='surface').mean()
check_std=full_data.groupby(by='surface').std()

check_mean=check_mean.T
for col in check_mean.columns:
    plt.figure()
    plt.plot(check_mean[col])
    plt.title(col)
    plt.xticks(rotation=45)
    plt.show()

#오일러 각으로 변환
def quaternion_to_euler(x,y,z,w):
    t0 = +2.0*(w*x + y*z)
    t1 = +1.0-2.0*(x*x + y*y)
    roll =np.arctan2(t0,t1)

    t2 = +2.0 * (w * y - z * x)
    t2[t2>+1.0] = +1.0
    t2[t2<-1.0] = -1.0
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return roll, yaw, pitch
#
for df in X_train_df, X_test_df :
    roll_, pitch_, yaw_ = quaternion_to_euler(df.orientation_X,df.orientation_Y,df.orientation_Z,df.orientation_W)
    df['roll']=roll_
    df['pitch']=pitch_
    df['yaw']=yaw_

mean_df=pd.DataFrame()
for col in ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X', 'linear_acceleration_Y','linear_acceleration_Z']:
    mean_df[col+'_mean']=X_train_df.groupby(by='series_id')[col].mean()
mean_df=mean_df.reset_index()
X_train_df=pd.merge(X_train_df,mean_df, on='series_id',how='inner')

mean_df=pd.DataFrame()
for col in ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X', 'linear_acceleration_Y','linear_acceleration_Z']:
    mean_df[col+'_mean']=X_test_df.groupby(by='series_id')[col].mean()
mean_df=mean_df.reset_index()
X_test_df=pd.merge(X_test_df,mean_df, on='series_id',how='inner')
X_test_df

std_df=pd.DataFrame()
for col in ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X', 'linear_acceleration_Y','linear_acceleration_Z']:
    std_df[col+'_std']=X_train_df.groupby(by='series_id')[col].std()
std_df=std_df.reset_index()
X_train_df=pd.merge(X_train_df,std_df, on='series_id',how='inner')

std_df=pd.DataFrame()
for col in ['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X', 'linear_acceleration_Y','linear_acceleration_Z']:
    std_df[col+'_std']=X_test_df.groupby(by='series_id')[col].std()
std_df=std_df.reset_index()
X_test_df=pd.merge(X_test_df,std_df, on='series_id',how='inner')
X_test_df

def create_X(df, nrows,ncols):
    X = np.zeros((nrows,128,ncols))
    X[:,:,0] = df['orientation_X'].values.reshape((-1,128))
    X[:,:,1] = df['orientation_Y'].values.reshape((-1,128))
    X[:,:,2] = df['orientation_W'].values.reshape((-1,128))
    X[:,:,3] = df['orientation_Z'].values.reshape((-1,128))

    X[:,:,4] = df['angular_velocity_X'].values.reshape((-1,128))
    X[:,:,5] = df['angular_velocity_Y'].values.reshape((-1,128))
    X[:,:,6] = df['angular_velocity_Z'].values.reshape((-1,128))

    X[:,:,7] = df['linear_acceleration_X'].values.reshape((-1,128))
    X[:,:,8] = df['linear_acceleration_Y'].values.reshape((-1,128))
    X[:,:,9] = df['linear_acceleration_Z'].values.reshape((-1,128))

    # X[:,:,10]= df['roll'].values.reshape((-1,128))
    # X[:,:,11]= df['pitch'].values.reshape((-1,128))
    # X[:,:,12]= df['yaw'].values.reshape((-1,128))
    #
    # X[:,:,10]= df['total_anglr_vel'].values.reshape((-1,128))
    # X[:,:,11]= df['total_linr_acc'].values.reshape((-1,128))
    # X[:,:,12]= df['total_xyz'].values.reshape((-1,128))
    # X[:,:,13]= df['acc_vs_vel'].values.reshape((-1,128))

    for i in range(ncols):
        X[:,:,[i]] = X[:,:,[i]]-X[:,:,[i]].mean()/X[:,:,[i]].std()

    # Detrending each signal
    from scipy import signal
    for i in range(ncols):
        X[:,:,i] = signal.detrend(X[:,:,i])
    return X

X_train_x = create_X(X_train_df, 3810,10)
X_test_x =create_X(X_test_df,3816,10)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train_df.columns
need_col=X_train_df.columns.difference(['row_id','measurement_number','group_id','series_id'])
X_train=X_train_df[need_col]
X_test=X_test_df[need_col]

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

X_train=X_train.reshape(-1,128,22)
X_test=X_test.reshape(-1,128,22)

X_train.shape
X_test.shape

encoder=LabelEncoder()
encoder.fit(y_train_df['surface'])
labels=encoder.transform(y_train_df['surface'])
y_categorical=to_categorical(labels)

from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64,return_sequences=True,input_shape=(128,22)),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(9,activation='softmax')
    ])

    model.compile(tf.keras.optimizers.Nadam(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

model=get_model()
model.summary()

p_val = np.zeros((X_train.shape[0], 9)) # train 예측값 담을 빈 행렬생성
p_tst = np.zeros((X_test.shape[0], 9))  # test 예측값 담을 빈 행렬생성

for i, (i_trn, i_val) in enumerate(cv.split(X_train, labels), 1):
    print(f'training model for CV #{i}')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3) #조기종료

    clf = get_model()
    clf.fit(X_train[i_trn],
            labels[i_trn],
            validation_data=(X_train[i_val], labels[i_val]),
            epochs=50,
            batch_size=20)
    p_val[i_val, :] = clf.predict(X_train[i_val])
    p_tst += clf.predict(X_test)/5


train_pred=pd.DataFrame()
train_pred['y']=np.argmax(y_categorical,axis=1)
train_pred['pred']=np.argmax(p_val,axis=1)
train_pred


from sklearn.metrics import classification_report
classification_report(train_pred['y'], train_pred['pred'])

sub=pd.read_csv('sample_submission.csv')
sub['surface']=p_tst
sub['surface'].value_counts()
sub.to_csv('submission(surface)8.csv',index=False)
