import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


# predict 실패한 이미지 출력
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

    # for i in range(x_train.shape[0]):
    #     count += 1
    #     if count == 10:
    #         plt.text(5,5,img_num)
    #         plt.show()
    #         fig = plt.figure()
    #
    #         count = 0
    #         img_num += 1
    #
    #     else :
    #         fig.add_subplot(3, 3, count)
    #         plt.imshow(x_train[i, :, :, :])
    #         plt.axis('off')
    #         plt.title(f'R:{y_train[i]}, P:{y_predicted[i]}')
    #         incorrect_lables.append(img_id[i])
    #
    #         incorrect_lables.writerow([img_id[i],y_train[i], y_predicted[i]])


    #
    # incorrect_df['id'] = incorrect_lables
    # incorrect_df['category'] = y_train
    # incorrect_df['predicted'] = y_predicted
    # incorrect_df.to_csv(f'incorrect.csv', index=False)


    incorrect_df['category'].value_counts()[:].plot(kind='bar')
    plt.title('incorrect category ( answer )')
    plt.show()

    incorrect_df['predicted'].value_counts()[:].plot(kind='bar')
    plt.title('incorrect prediction')
    plt.show()


# pca 분석 그래프
def show_graph(dataframe):

    plt.figure()

    pca = PCA(n_components=2)
    proj = pca.fit_transform(dataframe.data)
    plt.scatter(proj[:, 0], proj[:, 1], c=dataframe.target, cmap="Paired")
    plt.colorbar()

    plt.show()

    labels = [i for i in range(23)]
    fig = plt.figure(figsize=(8, 8))

    ax  = fig.add_subplot(1,1,1)
    ax.set_xlabel('Images', fontsize=15)
    ax.set_ylabel('Lables', fontsize=15)

    colors = ["#163d4e","#1f6642","#54792f","#a07949","#d07e93","#cf9cda",
              "#c1caf3","#d2eeef","#ffffff","#2b446e","#7f7c75","#948f78",
              "#23171b","#4a58dd","#2f9df5","#27d7c4","#4df884","#95fb51",
              "#dedd32","#ffa423","#f65f18","#ba2208","#900c00"]

    for label, color in tqdm(zip(labels, colors)):
        indicesToKeep = dataframe['Predicted'] == label
        ax.scatter(dataframe.loc[indicesToKeep, 'Id'],
                   dataframe.loc[indicesToKeep, 'Predicted'],
                   c = color,
                   s = 30)

    ax.legend(labels)
    ax.grid()

    print("plt show processing....")
    plt.show()

    fig, ax = plt.subplots()


def show_confusion(y_train, predictions):
    cm = confusion_matrix(y_train.argmax(axis=1), predictions.argmax(axis=1))
    df_cm = pd.DataFrame(cm, range(14), range(14))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
    plt.show()