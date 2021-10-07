EDA and Preprocessing

train = pd.read_csv("../input/how-much-did-it-rain-ii/train.zip",dtype=d)
train

train.keys()

train.loc[train["Id"] == 862571]

train.loc[train["Id"] == 5]

train.isna().sum()

pd.set_option('display.float_format', lambda x: '%.3f' % x)

train.fillna(0, inplace=True)
train[["minutes_past","radardist_km","Expected"]].describe()

calculate the correlation matrix

corr_mat = train.corr()
corr_mat.style.background_gradient(cmap='coolwarm')

improt mataplotlib.pyplot as plt

f = plt.figure(figsize(10, 10))
plt.matshow(corr_mat,fignum=f.number)
plt.colorbar()

plt.figure(figsize=(15, 10))
plt.hist(train["Expected"].unique())

train.grop(train[train['Expected'] >= 106].index,inplace=True)
train

plt.figure(figsize=(15, 10))
plt.scatter(np.arange(len(train["Expected"].unique())),train["Expected"].unique())

plt.figure(figsize=(15, 10))
plt.hist(train["Expected"].unique())

train_grouped = train.groupby('Id')
target = pd.DataFrame(train_grouped['Expected'].mean())

target.reset_index(inplace=True)
target = target["Expected"]
target

def pad_series(X,target_len=19)
    seq_len = X.shape[0]
    pad_size = target_len-seq_len
    if (pad_size > 0 ):
        X = np.pad(X,((0,pad_size),(0,0)),'constant',constant_values=0.)
        return X,  seq_len

INPUT_WIDTH = 19
data_size = len(train_grouped)
X_train = np.empty((data_size,INPUT_WIDTH,22))
seq_lengths = np.zeros(data_size)
y_train = np.zeros(data_size)

i = 0
for _,group in train_grouped:
    X = group.values
    seq_len = X.shape[0]
    X_train[i,:seq_len,:] = X[:,1:23]
    y_train[i] = X[0,23]
    i += 1
    del X
    
del train_grouped

Split the model into Train and Valid set

from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(X_train,target,random_state=42,shuffle=True)
Convert into tf.data.Dataset to avoid Out of memory while training, and delete unused variables.

import tensorflow as tf

del X_train
del target

train_data = tf.data.dataset.from_tensor_slices((x_train,y_train))
valid_data = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
train_data = train_data.batch(32)
valid_data = valid_data.batch(32)

del x_train
del x_valid
del y_train
del y_valid

Create and train the Model
import tensorflow as tf

def create_model(shape=(19,22)):
    tfkl = tf,keras.layers
    model = tf.keras.Sequentail([
        tfkl.Bidirectional((tfkl.LSTM(128,return_sequences=True)),input_shape=shape),
        tfkl.Bidirectional(tfkl.LSTM(64)),
        tfkl.Dense(64, activation="linear"),
        tfkl.Dense(1, activation="linear")
    ])
    
    model.compile(loss='mean_absolute_error',optimizer="adam")
    return model

model = create_model()
model.summary()

model.fit(train_data,epochs=100,validation_data=valid_data,
         callbacks=[ft.keras.callbacks.ReduceLROnPlateau(),
                   tf.keras.callbacks.EarlyStopping(patience = 10),
                    tf.keras.callbacks.ModelCheckpoint("model.h5",save_best_only=True)
                   ])

col_list.pop()
d = {c; np.float32 for c in col_list}

test = pd.read_csv("../input/how-much-did-it-rain-ii/test.zip",dtype=d)
test[test.columns[1:]] = test[test.columns[1:]].astype(np.float32)
test_ids = test['Id'].unique()

# Convert all NaNs to zero
test = test.reset_index(drop=True)
test.fillna(0.0,inplace=True)
test_groups = test.groupby('Id')
test_size = len(test_groups)

X_test = np.zeros((test,size,INPUT_WIDTH,22), dtype=np.float32)

i = 0 
for _,group in test_groups:
    X = group.values
    seq_len = X.shape[0]
    X_test[i,:seq_len,:] = X[:,1:23]
    i += 1
    del X
    
del test_groups
X_test.shape

submission = pd.read_csv("../input/how-much-did-it-rain-ii/sample_solution.csv.zip")
submission
