import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# TSNE
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import gc
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

debug = False
df = pd.read_json('../input/train.json').set_index('id')
test_df = pd.read_json('../input/test.json').set_index('id')
if debug is True:
    df = df.sample(100)
    test_df = test_df.sample(100)
traindex = df.index
testdex = test_df.index
print("Training Data Shape: ",df.shape)
print("Testing Data Shape: ", test_df.shape)
y = df.cuisine.copy()

# Combine For Pre-Processing
df = pd.concat([df.drop("cuisine", axis=1), test_df], axis=0)
df_index = df.index
print("All Data Shape: ", df.shape)
del test_df; gc.collect();

sns.countplot(y=y, order=y.value_counts().reset_index()["index"])
plt.title("Cuisine Distribution")
plt.show()

df.head()

print("Word Cloud Function..")
stopwords = set(STOPWORDS)
size = (20,10)

def cloud(text, title, stopwords=stopwords, size=size):
    """
    Function to plot WordCloud
    Includes: 
    """
    # Setting figure parameters
    mpl.rcParams['figure.figsize']=(10.0,10.0)
    mpl.rcParams['font.size']=12
    mpl.rcParams['savefig.dpi']=100
    mpl.rcParams['figure.subplot.bottom']=.1 
    
    # Processing Text
    # Redundant when combined with my Preprocessing function
    wordcloud = WordCloud(width=1600, height=800,
                          background_color='black',
                          stopwords=stopwords,
                         ).generate(str(text))
    
    # Output Visualization
    fig = plt.figure(figsize=size, dpi=80, facecolor='k',edgecolor='k')
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=50,color='y')
    plt.tight_layout(pad=0)
    plt.show()

# Data Set for Word Clouds
df["ing"] = df.ingredients.apply(lambda x: list(map(str, x)), 1).str.join(' ')
# All
cloud(df["ing"].values, title="All Cuisine", size=[8,5])

print("Cuisine WordClouds")
cloud_df = pd.concat([df.loc[traindex,'ing'], y],axis=1)
for cuisine_x in y.unique():
    cloud(cloud_df.loc[cloud_df.cuisine == cuisine_x, "ing"].values, title="{} Cuisine".format(cuisine_x.capitalize()), size=[8,5])
df.drop('ing',axis=1,inplace=True)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
dummies = vect.fit_transform(df['ingredients'].apply(','.join)) 

df = pd.DataFrame(dummies.todense(),columns=vect.get_feature_names())
print("Vocab Length: ", len(vect.get_feature_names()))
print("All Data Shape: ", df.shape)
df.index= df_index

print("Number of Predictors: ", df.shape[0])
df.head()

# Create the visualizer and draw the vectors
plt.figure(figsize = [15,9])
tsne = TSNEVisualizer()
tsne.fit(df.loc[traindex,:][:7000], y[:7000])
tsne.poof()

X = df.loc[traindex,:]
print("Number of Cuisine Types: ", y.nunique())
print("X Shape: ", X.shape)
test_df = df.loc[testdex,:]
print("Test DF Shape: ", test_df.shape)
del df; gc.collect();

LogisticRegression().get_params().keys()

model = LogisticRegression(multi_class= 'ovr')
score = cross_validate(model, X, y, return_train_score=False)
score["test_score"].mean()

model.fit(X,y)
submission = model.predict(test_df)
submission_df = pd.Series(submission, index=testdex).rename('cuisine')
submission_df.to_csv("submission.csv", index=True, header=True)
print(submission_df.head())