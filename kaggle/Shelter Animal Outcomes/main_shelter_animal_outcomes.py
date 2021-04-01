
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV


def label_age (row):
  if row['AgeuponOutcome'] == "0 years" :
      return 0
  if row['AgeuponOutcome'] == "1 year" :
      return 1
  if row['AgeuponOutcome'] == "2 years" :
      return 2
  if row['AgeuponOutcome'] == "3 years" :
      return 3
  if row['AgeuponOutcome'] == "4 years" :
      return 4
  if row['AgeuponOutcome'] == "5 years" :
      return 5
  if row['AgeuponOutcome'] == "6 years" :
      return 6
  if row['AgeuponOutcome'] == "7 years" :
      return 7
  if row['AgeuponOutcome'] == "8 years" :
      return 8
  if row['AgeuponOutcome'] == "9 years" :
      return 9
  if row['AgeuponOutcome'] == "10 years" :
      return 10
  if row['AgeuponOutcome'] == "11 years" :
      return 11
  if row['AgeuponOutcome'] == "12 years" :
      return 12
  if row['AgeuponOutcome'] == "13 years" :
      return 13
  if row['AgeuponOutcome'] == "14 years" :
      return 14
  if row['AgeuponOutcome'] == "15 years" :
      return 15
  if row['AgeuponOutcome'] == "16 years" :
      return 16
  if row['AgeuponOutcome'] == "17 years" :
      return 17
  if row['AgeuponOutcome'] == "18 years" :
      return 18
  if row['AgeuponOutcome'] == "20 years" :
      return 20
  if row['AgeuponOutcome'] == "1 month" :
      return 1/12
  if row['AgeuponOutcome'] == "2 months" :
      return 2/12
  if row['AgeuponOutcome'] == "3 months" :
      return 3/12
  if row['AgeuponOutcome'] == "4 months" :
      return 4/12
  if row['AgeuponOutcome'] == "5 months" :
      return 5/12
  if row['AgeuponOutcome'] == "6 months" :
      return 6/12
  if row['AgeuponOutcome'] == "7 months" :
      return 7/12
  if row['AgeuponOutcome'] == "8 months" :
      return 8/12
  if row['AgeuponOutcome'] == "9 months" :
      return 9/12
  if row['AgeuponOutcome'] == "10 months" :
      return 10/12
  if row['AgeuponOutcome'] == "11 months" :
      return 11/12
  if row['AgeuponOutcome'] == "1 week" :
      return 1/48
  if row['AgeuponOutcome'] == "1 weeks" :
      return 1/48
  if row['AgeuponOutcome'] == "2 weeks" :
      return 2/48
  if row['AgeuponOutcome'] == "3 weeks" :
      return 3/48
  if row['AgeuponOutcome'] == "4 weeks" :
      return 4/48
  if row['AgeuponOutcome'] == "5 weeks" :
      return 5/48
  if row['AgeuponOutcome'] == "1 day" :
      return 1/336
  if row['AgeuponOutcome'] == "2 days" :
      return 2/336
  if row['AgeuponOutcome'] == "3 days" :
      return 3/336
  if row['AgeuponOutcome'] == "4 days" :
      return 4/336
  if row['AgeuponOutcome'] == "5 days" :
      return 5/336
  if row['AgeuponOutcome'] == "6 days" :
      return 6/336


def munge(data, train):
    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0, "HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)
    data['AnimalType'] = data['AnimalType'].map({'Cat': 0, 'Dog': 1})

    if (train):
        data.drop(['AnimalID', 'OutcomeSubtype'], axis=1, inplace=True)
        data['OutcomeType'] = data['OutcomeType'].map(
            {'Return_to_owner': 4, 'Euthanasia': 3, 'Adoption': 0, 'Transfer': 5, 'Died': 2})

    gender = {'Neutered Male': 1, 'Spayed Female': 2, 'Intact Male': 3, 'Intact Female': 4, 'Unknown': 5, np.nan: 0}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)

    def agetodays(x):
        try:
            y = x.split()
        except:
            return None
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365 / 12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])


    data['AgeInDays'] = data['AgeuponOutcome'].map(agetodays)
    data.loc[(data['AgeInDays'].isnull()), 'AgeInDays'] = data['AgeInDays'].median()

    data['Year'] = data['DateTime'].str[:4].astype(int)
    data['Month'] = data['DateTime'].str[5:7].astype(int)
    data['Day'] = data['DateTime'].str[8:10].astype(int)
    data['Hour'] = data['DateTime'].str[11:13].astype(int)
    data['Minute'] = data['DateTime'].str[14:16].astype(int)

    data['Name+Gender'] = data['HasName'] + data['SexuponOutcome']
    data['Type+Gender'] = data['AnimalType'] + data['SexuponOutcome']
    data['IsMix'] = data['Breed'].str.contains('mix', case=False).astype(int)

    return data.drop(['AgeuponOutcome', 'Name', 'Breed', 'Color', 'DateTime'], axis=1)


def best_params(data):
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 150, 200],  # 400
        'max_depth': [None, 6, 9],
        'min_samples_split': [0.005, 0.01, 0.03],
        'max_features': ['auto', 'sqrt'],
    }

    kf = KFold(random_state=0,
               n_splits=5,
               shuffle=True,
               )

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=kf, n_jobs=-1)
    CV_rfc.fit(data[0::, 1::], data[0::, 0])
    print(CV_rfc.best_params_)
    print(CV_rfc.best_score_)
    return CV_rfc.best_params_


if __name__ == "__main__":
    in_file_train = '../235_ani/data/train.csv'
    in_file_test = '../235_ani/data/test.csv'

    print("Loading data...\n")
    pd_train = pd.read_csv(in_file_train)
    pd_test = pd.read_csv(in_file_test)

    print("Munging data...\n")
    pd_train = munge(pd_train, True)
    pd_test = munge(pd_test, False)

    pd_test.drop('ID', inplace=True, axis=1)

    train = pd_train.values
    test = pd_test.values

    print("Calculating best case params...\n")
    bestprm = best_params(train)
    print(bestprm)
    print(type(bestprm))
    print("Predicting... \n")
    forest = RandomForestClassifier(n_estimators=bestprm['n_estimators'], max_depth=bestprm['max_depth'],
                                    min_samples_split=bestprm['min_samples_split'],
                                    max_features=bestprm['max_features'])
    forest = forest.fit(train[0::, 1::], train[0::, 0])
    predictions = forest.predict_proba(test)

    output = pd.DataFrame(predictions, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    output.columns.names = ['ID']
    output.index.names = ['ID']
    output.index += 1

    print("Writing predictions.csv\n")

    print(output)

    output.to_csv('output.csv')

    print("Done.\n")



'''



def notebook():
    df = pd.read_csv("../235_ani/data/train.csv")

    # There is NaN and Unknown in Sex
    df["SexuponOutcome"] = df["SexuponOutcome"].fillna("Unknown")
    # Plot outcome based on sex
    sns.countplot(y="OutcomeType", data=df, hue="SexuponOutcome")
    # Time series
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.date
    plt.figure(figsize=(17, 6))
    df["OutcomeType"].groupby(df["DateTime"]).count().plot(kind="line")

    # For each outcome
    monthGroup = df["DateTime"].groupby(df["OutcomeType"])
    plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    plt.subplots_adjust(hspace=0.7)
    colors = list('rgbcmyk')
    for i, (_, g) in enumerate(monthGroup):
        plt.subplot(5, 1, i + 1)
        plt.title(_)
        g.groupby(df["DateTime"]).count().plot(kind="line", color=colors[i])

    # Monthly time series
    df_ym = df.DateTime.map(lambda x: x.strftime('%Y-%m'))
    df_ym_outcomeGroup = df_ym.groupby(df["OutcomeType"])

    plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    plt.subplots_adjust(hspace=0.7)
    colors = list('rgbcmyk')
    for i, (_, g) in enumerate(df_ym_outcomeGroup):
        plt.subplot(5, 1, i + 1)
        plt.title(_)
        g.groupby(df_ym).count().plot(kind="line", color=colors[i])

    # For each outcome
    df_dow = pd.to_datetime(df.DateTime).dt.dayofweek
    dayinweekGroup = df["DateTime"].groupby(df["OutcomeType"])
    plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    plt.subplots_adjust(hspace=0.7)
    colors = list('rgbcmyk')
    for i, (_, g) in enumerate(dayinweekGroup):
        plt.subplot(5, 1, i + 1)
        plt.title(_)
        g.groupby(df_dow).count().plot(kind="line", color=colors[i])

    # Monthly time series
    df_ym = df.DateTime.map(lambda x: x.strftime('%d'))
    df_ym_outcomeGroup = df_ym.groupby(df["OutcomeType"])

    plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    plt.subplots_adjust(hspace=0.7)
    colors = list('rgbcmyk')
    for i, (_, g) in enumerate(df_ym_outcomeGroup):
        plt.subplot(5, 1, i + 1)
        plt.title(_)
        g.groupby(df_ym).count().plot(kind="line", color=colors[i])

    # Convert age string to be float
    df["Agecat"] = df.apply(lambda row: label_age(row), axis=1)
    df_age_outcomeGroup = df["Agecat"].groupby(df["OutcomeType"])

    plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    plt.subplots_adjust(hspace=0.7)
    colors = list('rgbcmyk')
    for i, (_, g) in enumerate(df_age_outcomeGroup):
        ax = plt.subplot(5, 1, i + 1)
        ax.set_xscale('log')
        plt.title(_)
        g.groupby(df["Agecat"]).count().plot(kind="line", color=colors[i])

    # Prepare for training data
    ytrain = df["OutcomeType"]
    Xtrain = df.drop(["OutcomeType", "OutcomeSubtype", "AgeuponOutcome", "AnimalID", "Name"], axis=1)
    Xtrain.head()

    le_anima = preprocessing.LabelEncoder()
    Xtrain.AnimalType = le_anima.fit_transform(Xtrain.AnimalType)
    le_sex = preprocessing.LabelEncoder()
    Xtrain.SexuponOutcome = le_sex.fit_transform(Xtrain.SexuponOutcome)
    le_breed = preprocessing.LabelEncoder()
    Xtrain.Breed = le_breed.fit_transform(Xtrain.Breed)
    le_color = preprocessing.LabelEncoder()
    Xtrain.Color = le_color.fit_transform(Xtrain.Color)
    le_out = preprocessing.LabelEncoder()
    ytrain = le_out.fit_transform(ytrain)
    # Let's see
    Xtrain.head()

    ## Explode date time
    xdt = pd.to_datetime(Xtrain.DateTime)
    Xtrain["dow"] = xdt.dt.dayofweek
    Xtrain["month"] = xdt.dt.month
    Xtrain["year"] = xdt.dt.year

    Xtrain = Xtrain.drop(["DateTime"], axis=1)

    Xtrain = Xtrain.fillna(-1)

    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(Xtrain, ytrain)
    # Let's see the train accuracy
    tra_score = rf.score(Xtrain, ytrain)
    print("Training accuracy ", tra_score)

    m = [6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]
    train_err = []
    val_err = []
    perm = np.random.permutation(len(Xtrain))
    Xtr = Xtrain.iloc[perm[0:20000]]
    ytr = ytrain[perm[0:20000]]
    Xval = Xtrain.iloc[perm[20001:]]
    yval = ytrain[perm[20001:]]
    for i in range(8):
        trainSize = m[i]
        perm = np.random.permutation(len(Xtr))
        XtrNow = Xtr.iloc[perm[0:trainSize]]
        ytrNow = ytr[perm[0:trainSize]]
        # Do random forest
        rf = RandomForestClassifier(n_estimators=1000)
        rf.fit(XtrNow, ytrNow)
        # Let's see the train accuracy
        rScore = rf.score(XtrNow, ytrNow)
        vScore = rf.score(Xval, yval)
        train_err.append(rScore)
        val_err.append(vScore)
    plt.plot(m, train_err, 'r', m, val_err, 'b')

'''


'''
    kf = KFold(n_splits=10, shuffle=True)
    print(kf)
    print(type(kf))

    foldScore = []
    for train_index, test_index in kf.split(rf):
        Xtr, X_test = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr, y_test = ytrain[train_index], ytrain[test_index]
        rf.fit(Xtr, ytr)
        val_acc = rf.score(X_test, y_test)
        tra_acc = rf.score(Xtr, ytr)
        print(val_acc, tra_acc)
        foldScore.append([val_acc, tra_acc])
    print("Over all Fold", np.mean(foldScore, axis=0))
'''


'''
    # Get test data
    tt = pd.read_csv("../235_ani/data/test.csv")
    tt.head()

    IDtest = tt["ID"]
    Xtest = tt.drop(["ID", "Name"], axis=1)

    Xtest.AnimalType = le_anima.fit_transform(Xtest.AnimalType)
    Xtest.SexuponOutcome = le_sex.fit_transform(Xtest.SexuponOutcome)
    Xtest.Breed = le_breed.fit_transform(Xtest.Breed)
    Xtest.Color = le_color.fit_transform(Xtest.Color)
    Xtest["AgeCat"] = Xtest.apply(lambda row: label_age(row), axis=1)

    xtt = pd.to_datetime(Xtest.DateTime)
    Xtest["dow"] = xdt.dt.dayofweek
    Xtest["month"] = xdt.dt.month
    Xtest["year"] = xdt.dt.year
    Xtest = Xtest.drop(["AgeuponOutcome", "DateTime"], axis=1)
    Xtest.head()

    Xtest = Xtest.fillna(-1)

    # Get prediction
    ytest = rf.predict(Xtest)
    ytestproba = rf.predict_proba(Xtest)
    yfin = le_out.inverse_transform(ytest)

    yprint = pd.DataFrame()
    yprint["ID"] = IDtest
    yprint["Adoption"] = (yfin == "Adoption").astype(int)
    yprint["Died"] = (yfin == "Died").astype(int)
    yprint["Euthanasia"] = (yfin == "Euthanasia").astype(int)
    yprint["Return_to_owner"] = (yfin == "Return_to_owner").astype(int)
    yprint["Transfer"] = (yfin == "Transfer").astype(int)

    yprint.to_csv("submit_randomforest.csv", index=False)

    f = open("submit_rfproba.csv", "w")
    filewrite = csv.writer(f)
    filewrite.writerow(["ID", "Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"])
    for i in range(len(ytestproba)):
        filewrite.writerow(
            [IDtest[i], ytestproba[i, 0], ytestproba[i, 1], ytestproba[i, 2], ytestproba[i, 3], ytestproba[i, 4]])
    f.close()

    # Let us try again but without using Breed and Color feature
    Xtrain2 = Xtrain.drop(["Breed", "Color"], axis=1)
    rf2 = RandomForestClassifier(n_estimators=1000)
    rf2.fit(Xtrain2, ytrain)

    # How is our training accuracy now?
    yt_pred2 = rf2.predict(Xtrain2)
    np.mean(ytrain == yt_pred2)

    # Get prediction and print!
    Xtest2 = Xtest.drop(["Breed", "Color"], axis=1);
    ytest2 = rf2.predict(Xtest2)
    ytestproba2 = rf2.predict_proba(Xtest2)
    yfin2 = le_out.inverse_transform(ytest2)

    f = open("submit_rfproba2.csv", "w")
    filewrite = csv.writer(f)
    filewrite.writerow(["ID", "Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"])
    for i in range(len(ytestproba2)):
        filewrite.writerow(
            [IDtest[i], ytestproba2[i, 0], ytestproba2[i, 1], ytestproba2[i, 2], ytestproba2[i, 3], ytestproba2[i, 4]])

    f.close()




def read(csv_dir):
    csv_file = pd.read_csv(csv_dir)
    csv_file = csv_file.fillna(0)
    return csv_file

def remove_col(file):
    temp = pd.DataFrame(file)
    temp = temp.iloc[:,3:7]
    return temp

def remove_col2(file):

    temp = pd.DataFrame(file)
    temp = temp.iloc[:,3:5]
    return temp

def changeAni(file):
    result = pd.DataFrame([])
    for i in file['AnimalType']:
        if i =='Dog':
            result = result.append([1])
        else:
            result = result.append([2])
    return result

def changeSex(file):
    result = pd.DataFrame([])
    for i in file['SexuponOutcome']:
        if i =='Neutered Male':
            result = result.append([1])
        elif i == 'Intact Female':
            result = result.append([2])
        elif i == 'Spayed Female':
            result = result.append([3])
        elif i == 'Unknown':
            result = result.append([4])
        elif i == 'Intact Male':
            result = result.append([5])
        else:
            result = result.append([0])
    return result

def changeOut(file):
    result = pd.DataFrame([])
    for i in file['OutcomeType']:
        if i =='Euthanasia':
            result = result.append([1])
        elif i == 'Transfer':
            result = result.append([2])
        elif i == 'Return_to_owner':
            result = result.append([3])
        elif i == 'Died':
            result = result.append([4])
        elif i == 'Intact Male':
            result = result.append([5])
        else:
            result = result.append([1])
    return result

def lr(x_train,y,x_test): #x->2d, y->1d
    reg = LinearRegression()
    reg.fit(x_train,y)
    y_predict = reg.predict(x_test)
    return y_predict

def min_max(file):
    result = pd.DataFrame([])
    for i in file:
        temp = i
        if temp <=1:
            temp = 1
        elif temp >=5:
            temp = 5
        elif temp >=4 and temp<5:
            temp =4
        elif temp >=3 and temp<4:
            temp =3
        elif temp >=2 and temp<3:
            temp =2
        elif temp >=1 and temp<2:
            temp =1
        result = result.append([temp])
    return result

def rearrange(y_predict):
    #temp = [[],[],[],[],[],[]]
    #result = pd.DataFrame(temp,columns=['ID','Adoption','Died',"Euthanasia",'Return_to_owner','Transfer'])
    ID = []#pd.DataFrame([])
    Adoption = []#pd.DataFrame([])
    Died = []#pd.DataFrame([])
    Euthanasia =[]#pd.DataFrame([])
    Return_to_owner =[]#pd.DataFrame([])
    Transfer =[]#pd.DataFrame([])
    for i in range(len(y_predict)):
        ID.append(i)
        Adoption.append(0)
        Died.append(1)
        Euthanasia.append(0)
        Return_to_owner.append(0)
        Transfer.append(0)
    return ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer

def write_csv(ID, Adoption, Died, Euthanasia, Return_to_owner,Transfer):
    result = []
    print(ID)
    for i in range(len(ID)):
        result.append([ID[i]+1,Adoption[i],Died[i],Euthanasia[i],Return_to_owner[i],Transfer[i]])

    with open('submit.csv','w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        first = ['ID', 'Adoption','Died','Euthanasia','Return_to_owner','Transfer']
        writer.writerow(first)
        for i in range(len(result)):
            writer.writerow(result[i])


if __name__ == "__main__":
   main()
'''
'''
    train_x_ori = read('../235_ani/data/train.csv')
    test_x_ori  = read('../235_ani/data/test.csv' )
    train_x_col = remove_col(train_x_ori)
    test_x_col  = remove_col2(test_x_ori)
    train_ani = changeAni(train_x_col)
    train_sex = changeSex(train_x_col)
    train_out = changeOut(train_x_col)
    test_ani = changeAni(test_x_col)
    test_sex = changeSex(test_x_col)

    #result = lr(train_x,train_y, test_x)

    train_ani['sex'] = train_sex
    test_ani['sex'] = test_sex #1~11456
    y_predict = lr(train_ani,train_out,test_ani)
    y_predict = min_max(y_predict)
    ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer = rearrange(y_predict)
    write_csv(ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer)
'''
