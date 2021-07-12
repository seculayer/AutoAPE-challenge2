import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2, random, time, shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
%matplotlib inline 

import keras
from keras import backend
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

train_zip_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip'
test_zip_dir = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip'
extract_dir = '/kaggle/working/extracted_data'
train_dir = '/kaggle/working/train'
test_dir = '/kaggle/working/test'
os.makedirs(train_dir+'/dog', exist_ok=True)

import zipfile
with zipfile.ZipFile(train_zip_dir, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

with zipfile.ZipFile(test_zip_dir, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

def move_files(train_path,test_path):
    print('Moving Training Files ..')
    time.sleep(1)
    for i in tqdm(os.listdir(train_path)):        
        if 'dog' in i:
            shutil.copyfile(train_path+i,train_dir+'/dog/'+i )
        elif 'cat' in i:
            shutil.copyfile(train_path+i,train_dir+'/cat/'+i )
        else:
            print('unkown File', i)
            
    print('Moving Testing Files ..')
    time.sleep(1)
    for i in tqdm(os.listdir(test_path)):                
        shutil.copyfile(test_path+i, test_dir+'/test_data/'+i)
    #Delete original data    
    shutil.rmtree(extract_dir)

def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])
#Setting Image and model parameters
Image_width,Image_height = 299,299
batch_size=64
total_samples = get_num_files(train_dir)
val_split=0.2
n_train=total_samples*(1-val_split)
n_val=total_samples*val_split
num_classes = 2

train_image_gen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,validation_split=val_split)
#Data loader to load each batch on the RAM at each step.
train_generator = train_image_gen.flow_from_directory(train_dir,target_size=(Image_width,Image_height),
                                                      batch_size=batch_size,seed=42,subset='training',
                                                      shuffle=True,class_mode='categorical')
val_generator = train_image_gen.flow_from_directory(train_dir,target_size=(Image_width,Image_height),
                                                    batch_size=batch_size,seed=42,subset='validation',
                                                    shuffle=True,class_mode='categorical')
print(n_train,n_val)



        
move_files(extract_dir+'/train/', extract_dir+'/test/')

from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
#Prepare call backs
LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=2, factor=.5, min_lr=.00001)
EarlyStop_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
my_callback=[EarlyStop_callback, LR_callback]
#Prepare the model.
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)
x = InceptionV3_base_model.output
x_pool = GlobalAveragePooling2D()(x)

layer_to_Freeze=276    
for layer in model.layers[:layer_to_Freeze]:
    layer.trainable =False
for layer in model.layers[layer_to_Freeze:]:
    layer.trainable=True

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
#Training ...
history_transfer_learning = model.fit_generator(train_generator,epochs=30,
                                                steps_per_epoch=n_train//batch_size,
                                                validation_data=val_generator,
                                                validation_steps=n_val//batch_size,
                                                verbose=1,
                                                callbacks=my_callback,
                                                class_weight='auto')
final_pred = Dense(num_classes,activation='softmax')(x_pool)
model = Model(inputs=InceptionV3_base_model.input,outputs=final_pred)

score = model.evaluate_generator(val_generator,verbose=1)
print('Test loss: ', score[0])
print('Test accuracy', score[1])test_image_gen = ImageDataGenerator(rescale=1/255)
test_generator = test_image_gen.flow_from_directory(test_dir,target_size=(Image_width,Image_height),batch_size=1,seed=42,class_mode=None,shuffle=False)

y_pred = model.predict_generator(generator=test_generator,verbose=1)

submission = pd.DataFrame({'id':pd.Series(test_generator.filenames),'label':pd.Series(y_pred.clip(min=0.02,max=0.98)[:,1])})
submission['id'] = submission.id.str.extract('(\d+)')
submission['id']=pd.to_numeric(submission['id'])
submission.to_csv("MySubmission.csv",index=False)
#submission.nunique(axis=0)
submission.head(10)
os.makedirs(train_dir+'/cat', exist_ok=True)
os.makedirs(test_dir+'/test_data', exist_ok=True)