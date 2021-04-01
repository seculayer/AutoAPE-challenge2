
import os
from keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.densenet import DenseNet121 


def CNNdensenet(weights=None):
    model = DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None,
                        classes=1000)
    return model

def add_image_path(x):
    image_path="../smile/data/train/"+x
    temp_path="../smile/data/train/F0002/MID1/P00017_face3.jpg"
    #print(image_path)
    #return image_path

    if os.path.exists(image_path):
        #print(os.listdir(image_path)[0])
        path=os.path.join(image_path,os.listdir(image_path)[0])
        #print(path)
        return path
    else:
        return temp_path

def load_img(PATH):
    return np.array(Image.open(PATH))


def distance(x, y):
    return np.linalg.norm(x - y)

if __name__ == "__main__":
    train_df = pd.read_csv("../smile/data/train_relationships.csv")
    number = sorted(os.listdir("../smile/data/train/"))#train-faces

    members = {i: sorted(os.listdir(("../smile/data/train/") + i)) for i in number}

    test_path = "../smile/data/test"
    test_images_name = os.listdir(test_path)

    # To convert images into a matrix
    test_images = np.array([load_img(os.path.join(test_path, image)) for image in test_images_name])

    CNN = CNNdensenet()

    # To check results with the train set:

    im = Image.open('../smile/data/train/F1000/MID4/P10582_face1.jpg')
    im = np.array(im).astype(np.float32)
    im2 = Image.open('../smile/data/train/F1000/MID4/P10582_face1.jpg')
    im2 = np.array(im2).astype(np.float32)

    im = np.expand_dims(im, axis=0)

    im2 = np.expand_dims(im2, axis=0)
    np.concatenate([im, im2]).shape

    out = CNN.predict(np.concatenate([im, im2]))


    test_images = os.listdir(test_path)
    test = np.array([load_img(os.path.join(test_path, i)) for i in test_images])
    test_emb = CNN.predict(test)

    image_index = {imagen_numero: idx for idx, imagen_numero in enumerate(test_images)}
    submission = pd.read_csv('../smile/data/sample_submission.csv')

    splitting = [i.split('-') for i in submission.img_pair]

    distances = []
    for i in splitting:
        a = i[0]
        b = i[1]
        dist = distance(test_emb[image_index[a]], test_emb[image_index[b]])
        distances.append(dist)

    distances = np.array(distances) / np.max(distances)
    probability = 1 - ((distances / np.max(distances)))  # (0.2 is a security coeffcient)

    submission.is_related = probability
    submission.to_csv('submission.csv', index=False)

