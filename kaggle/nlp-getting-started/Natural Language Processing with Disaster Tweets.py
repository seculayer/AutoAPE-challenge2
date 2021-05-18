import random
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from time import time

#load dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

#finding missing values
missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values ':missing_values,
    'percent_missing %':percent_missing
}

##Remove redundant samples
train=train.drop_duplicates(subset=['text', 'target'], keep='first')

# Numbers of word for each sapmle in train & test data
train['text_length'] = train.text.apply(lambda x: len(x.split()))
test['text_length'] = test.text.apply(lambda x: len(x.split()))


# Data Cleaning#
pstem = PorterStemmer()
def clean_text(text):
    text= text.lower()
    text= re.sub('[0-9]', '', text)
    text  = "".join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens=[pstem.stem(word) for word in tokens]
    tokens=[word for word in tokens if word not in stopwords.words('english')]
    text = ' '.join(tokens)
    return text

train["clean"]=train["text"].apply(clean_text)
test["clean"]=test["text"].apply(clean_text)

# collecting all words in single list
list_= []
for i in train.clean:
    list_ += i
list_= ''.join(list_)
allWords=list_.split()
vocabulary= set(allWords)
len(vocabulary)

tfidf = TfidfVectorizer(sublinear_tf=True,max_features=60000, min_df=1, norm='l2',  ngram_range=(1,2))
features = tfidf.fit_transform(train.clean).toarray()
features.shape

features_test = tfidf.transform(test.clean).toarray()


#LogisticRegression
#split data into 4 parts with same distribution of classes.
skf = StratifiedKFold(n_splits=4, random_state=48, shuffle=True)
accuracy=[] # list contains the accuracy for each fold
n=1
y=train['target']

for trn_idx, test_idx in skf.split(features, y):
  start_time = time()
  X_tr,X_val=features[trn_idx],features[test_idx]
  y_tr,y_val=y.iloc[trn_idx],y.iloc[test_idx]
  model= LogisticRegression(max_iter=1000,C=3)
  
  model.fit(X_tr,y_tr)
  s = model.predict(X_val)
  sub[str(n)]= model.predict(features_test) 
  
  accuracy.append(accuracy_score(y_val, s))
  print((time() - start_time)/60,accuracy[n-1])
n+=1

# Evaluation
accuracy
np.mean(accuracy)*100

from sklearn.metrics import confusion_matrix, classification_report
pred_valid_y = model.predict(X_val)
print(classification_report(y_val, pred_valid_y ))


#Submission
df=sub[['1','2','3','4']].mode(axis=1)
sub['target']=df[0]    
sub=sub[['id','target']]
sub['target']=sub['target'].apply(lambda x : int(x))

sub.to_csv('submission.csv',index=False)

