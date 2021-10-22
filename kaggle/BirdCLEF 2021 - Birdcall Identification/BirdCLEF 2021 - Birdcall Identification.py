import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
import IPython.display as display

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.applications import VGG19, VGG16, ResNet50

import warnings
warnings.filterwarnings("ignore")
path = '/kaggle/input/birdclef-2021/'
os.listdir(path)


def read_ogg_file(path, file):
    """ Read ogg audio file and return numpay array and samplerate"""

    data, samplerate = sf.read(path + file)
    return data, samplerate


def plot_audio_file(data, samplerate):
    """ Plot the audio data"""

    sr = samplerate
    fig = plt.figure(figsize=(8, 4))
    x = range(len(data))
    y = data
    plt.plot(x, y)
    plt.plot(x, y, color='red')
    plt.legend(loc='upper center')
    plt.grid()


def plot_spectrogram(data, samplerate):
    """ Plot spectrogram with mel scaling """

    sr = samplerate
    spectrogram = librosa.feature.melspectrogram(data, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')


train_labels = pd.read_csv(path + 'train_soundscape_labels.csv')
train_meta = pd.read_csv(path + 'train_metadata.csv')
test_data = pd.read_csv(path + 'test.csv')
samp_subm = pd.read_csv(path + 'sample_submission.csv')
print('Number train label samples:', len(train_labels))
print('Number train meta samples:', len(train_meta))
print('Number train short folder:', len(os.listdir(path + 'train_short_audio')))
print('Number train audios:', len(os.listdir(path + 'train_soundscapes')))
print('Number test samples:', len(test_data))

row = 0
train_meta.iloc[row]

label = train_meta.loc[row, 'primary_label']
filename = train_meta.loc[row, 'filename']

# Check if the file is in the folder
filename in os.listdir(path+'train_short_audio/'+label)

data, samplerate = sf.read(path+'train_short_audio/'+label+'/'+filename)
print(data[:8])
print(samplerate)

train_labels['audio_id'].unique()

train_labels.groupby(by=['audio_id']).count()['birds'][:4]

print('original label:', train_labels.loc[458, 'birds'])
print('split into list:', train_labels.loc[458, 'birds'].split(' '))


labels = []
for row in train_labels.index:
    labels.extend(train_labels.loc[row, 'birds'].split(' '))
labels = list(set(labels))

print('Number of unique bird labels:', len(labels))

df_labels_train = pd.DataFrame(index=train_labels.index, columns=labels)
for row in train_labels.index:
    birds = train_labels.loc[row, 'birds'].split(' ')
    for bird in birds:
        df_labels_train.loc[row, bird] = 1
df_labels_train.fillna(0, inplace=True)

# We set a dummy value for the target label in the test data because we will need for the Data Generator
test_data['birds'] = 'nocall'

df_labels_test = pd.DataFrame(index=test_data.index, columns=labels)
for row in test_data.index:
    birds = test_data.loc[row, 'birds'].split(' ')
    for bird in birds:
        df_labels_test.loc[row, bird] = 1
df_labels_test.fillna(0, inplace=True)
df_labels_train.sum().sort_values(ascending=False)[:10]


train_labels = pd.concat([train_labels, df_labels_train], axis=1)
test_data = pd.concat([test_data, df_labels_test], axis=1)
file = os.listdir(path+'train_soundscapes')[0]

data, samplerate = read_ogg_file(path+'train_soundscapes/', file)
audio_id = file.split('_')[0]
site = file.split('_')[1]
print('audio_id:', audio_id, ', site:', site)

train_labels[(train_labels['audio_id']==int(audio_id)) & (train_labels['site']==site) & (train_labels['birds']!='nocall')]

sub_data = data[int(455 / 5) * 160000:int(460 / 5) * 160000]
data_lenght = 160000
audio_lenght = 5
num_labels = len(labels)
batch_size = 16
list_IDs_train, list_IDs_val = train_test_split(list(train_labels.index), test_size=0.33, random_state=2021)
list_IDs_test = list(samp_subm.index)


class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, data, batch_size):
        self.path = path
        self.list_IDs = list_IDs
        self.data = data
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        len_ = int(len(self.list_IDs) / self.batch_size)
        if len_ * self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        X = X.reshape((self.batch_size, 100, 1600 // 2))
        return X, y

    def __data_generation(self, list_IDs_temp):
        X = np.zeros((self.batch_size, data_lenght // 2))
        y = np.zeros((self.batch_size, num_labels))
        for i, ID in enumerate(list_IDs_temp):
            prefix = str(self.data.loc[ID, 'audio_id']) + '_' + self.data.loc[ID, 'site']
            file_list = [s for s in os.listdir(self.path) if prefix in s]
            if len(file_list) == 0:
                # Dummy for missing test audio files
                audio_file_fft = np.zeros((data_lenght // 2))
            else:
                file = file_list[0]  # [s for s in os.listdir(self.path) if prefix in s][0]
                audio_file, audio_sr = read_ogg_file(self.path, file)
                audio_file = audio_file[int((self.data.loc[ID, 'seconds'] - 5) / audio_lenght) * data_lenght:int(
                    self.data.loc[ID, 'seconds'] / audio_lenght) * data_lenght]
                audio_file_fft = np.abs(np.fft.fft(audio_file)[: len(audio_file) // 2])
                # scale data
                audio_file_fft = (audio_file_fft - audio_file_fft.mean()) / audio_file_fft.std()
            X[i,] = audio_file_fft
            y[i,] = self.data.loc[ID, self.data.columns[5:]].values
        return X, y


train_generator = DataGenerator(path + 'train_soundscapes/', list_IDs_train, train_labels, batch_size)
val_generator = DataGenerator(path + 'train_soundscapes/', list_IDs_val, train_labels, batch_size)
test_generator = DataGenerator(path + 'test_soundscapes/', list_IDs_test, test_data, batch_size)
epochs = 2
lernrate = 2e-3

model = Sequential()
model.add(Conv1D(64, input_shape=(100, 1600 // 2,), kernel_size=5, strides=4, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=(4)))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_labels, activation='sigmoid'))

model.compile(optimizer=Adam(lr=lernrate),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.summary()

history = model.fit_generator(generator=train_generator, validation_data=val_generator, epochs = epochs, workers=4)

y_pred = model.predict_generator(test_generator, verbose=1)

y_test = np.where(y_pred > 0.5, 1, 0)
for row in samp_subm.index:
    string = ''
    for col in range(len(y_test[row])):
        if y_test[row][col] == 1:
            if string == '':
                string += labels[col]
            else:
                string += ' ' + labels[col]
    if string == '':
        string = 'nocall'
    samp_subm.loc[row, 'birds'] = string
output = samp_subm
output.to_csv('submission.csv', index=False)