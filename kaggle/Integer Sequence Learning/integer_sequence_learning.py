import pickle
import pandas as pd
import numpy as np
import csv
from sklearn import linear_model


def preproc(dat = [], seq = ','):
    return dat.map(lambda x: np.array(x.split(seq), dtype = float))

if __name__ == "__main__":
    train_csv = '../k_integer_sequence_learning/input/train.csv'
    test_csv  = '../k_integer_sequence_learning/input/test.csv'
    train_dat = pd.read_csv(train_csv)
    test_dat = pd.read_csv(test_csv)
    print(train_dat.shape)
    frame = ['id', 'seq']
    train_dat.columns = frame
    test_dat.columns = frame
    train_dat['seq'] = preproc(train_dat['seq'])
    test_dat ['seq'] = preproc(test_dat ['seq'])###전처리완료

    x = np.array([])
    y = np.array([])

    y = pd.DataFrame({'y':y})
    for i in range(100):

        x_ = 0
        for j in range(len(train_dat['seq'][i])):
            try:
                temp = np.array(x_)
                x = np.append(x, temp)
                y = np.append(y, round(train_dat['seq'][i][j]))
                x_ += 1
            except BufferError as e:
                print(e)
                temp = np.array(x_)
                x = np.append(x,temp)
                y = np.append(y, 0)
                x_ += 1


    print("refining done")
    x =pd.DataFrame({'x':x})

    regressing = linear_model.LinearRegression()
    regressing.fit(x,y)

    test_x =[]
    #test_x.iloc[[0], :]
    #test_x.head(0)
    for i in test_dat['seq']:
        temp2 = [len(i)]
        test_x.append(len(i))
    test_x = pd.DataFrame(test_x)
    predict = regressing.predict(test_x)
    print(predict)
    print(type(predict))
    print(len(predict))

    with open('model.p', 'wb') as file:
        pickle.dump(regressing, file)

    temparr = [test_dat['id'],predict]
    temparrlen =len(temparr[1])


    result = []
    for i in range(len(train_dat['seq'])):
        result.append([test_dat['id'][i],round(predict[i])])



    with open('submit.csv','w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        first = ['Id', 'Last']
        writer.writerow(first)
        for i in range(len(result)):
            writer.writerow(result[i])
