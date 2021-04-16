import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


#  훈련 후 predict 실패한 이미지 출력
def incorrect_lables (x_train, y_train, y_predicted, label_df):
    fig = plt.figure()

    incorrect_idx = (y_train!=y_predicted)

    y_train = y_train[incorrect_idx]
    y_predicted = y_predicted[incorrect_idx]
    x_train = x_train[y_predicted, :, :, :]

    img_id = label_df['id']
    incorrect_lables = img_id[incorrect_idx]

    count = 0
    img_num = 1

    incorrect_df = pd.read_csv('./incorrect.csv')
    for i in range(len(incorrect_lables)):
        print(f'find incorrect lables..... {i} ')
        incorrect_df.loc[i] = [img_id[i], y_train[i], y_predicted[i]]

    incorrect_df.to_csv(f'incorrect_result.csv', index=False)


    incorrect_df['category'].value_counts()[:].plot(kind='bar')



#  confusion matrix 출력
def show_confusion(y_train, predictions):
    cm = confusion_matrix(y_train.argmax(axis=1), predictions.argmax(axis=1))
    df_cm = pd.DataFrame(cm, range(14), range(14))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
    plt.show()