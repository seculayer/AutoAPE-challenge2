from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Input, concatenate, Dropout, Conv1D, BatchNormalization, Bidirectional, LSTM
from tensorflow.keras.layers import Flatten, AveragePooling1D, GlobalMaxPooling1D, SpatialDropout1D, Activation, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from gensim.models.keyedvectors import KeyedVectors

train_data = pd.read_csv('../input/google-quest-challenge/train.csv')
test_data = pd.read_csv('../input/google-quest-challenge/test.csv')

#학습 결과 행 추출
sample_submission = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
class_names = list(sample_submission.columns[1:])

class_question = class_names[:21]
class_answer = class_names[21:]
question_output = train_data[class_question]
answer_output = train_data[class_answer]
output = train_data[class_names]

le = LabelEncoder()
categoria = train_data.category

## category 데이터를 라벨 숫자로 변환
train_categoria = le.fit_transform(categoria)
## 변환된 라벨을 원핫 인코딩
encode_category = tf.keras.utils.to_categorical(train_categoria)

## question_title, question_body, answer 행 추출
x = train_data.columns[[1,2,5]]
x = train_data[x]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(x.question_title)
tokenizer.fit_on_texts(x.question_body)
tokenizer.fit_on_texts(x.answer)

vocab_size = len(tokenizer.word_index) + 1

train_title = tokenizer.texts_to_sequences(x.question_title)
train_body = tokenizer.texts_to_sequences(x.question_body)
train_answer = tokenizer.texts_to_sequences(x.answer)

word_index = tokenizer.word_index

train_title_pad = pad_sequences(train_title)
train_body_pad = pad_sequences(train_body)
train_answer_pad = pad_sequences(train_answer)

embedding_dict = {}
f = open('../input/qna-w2v-min1-lower/qna_w2v_min1_lower', encoding='utf-8')
for line in f:
    word_vector = line.rsplit()
    word = word_vector[0]
    word_vector_arr = np.asarray(word_vector[1:], dtype='float32')
    embedding_dict[word] = word_vector_arr
f.close()

embedding_matrix = np.zeros((vocab_size, 100))

for word, i in tokenizer.word_index.items():
    temp = embedding_dict.get(word)

    if temp is not None:
        embedding_matrix[i] = temp


embedding = Embedding(
    *embedding_matrix.shape, ## 54533 100
    weights=[embedding_matrix],
    trainable=False,
    ## 0으로 패딩된 값은 무시
    mask_zero=True
)

### 제목 layer
title_input = Input(shape=(None, ))
embedded_title = embedding(title_input)


embedded_title1 = Conv1D(64, 5, activation='relu', strides=1)(embedded_title)
embedded_title1 = SpatialDropout1D(0.2)(embedded_title1)
embedded_title1 = GlobalMaxPooling1D()(embedded_title1)
embedded_title1 = Flatten()(embedded_title1)

embedded_title2 = Conv1D(128, 5, activation='relu', strides=1)(embedded_title)
embedded_title2 = SpatialDropout1D(0.2)(embedded_title2)
embedded_title2 = GlobalMaxPooling1D()(embedded_title2)
embedded_title2 = Flatten()(embedded_title2)

embedded_title3 = Conv1D(256, 5, activation='relu', strides=1)(embedded_title)
embedded_title3 = SpatialDropout1D(0.2)(embedded_title3)
embedded_title3 = GlobalMaxPooling1D()(embedded_title3)
embedded_title3 = Flatten()(embedded_title3)

### 질문 layer
question_input = Input(shape=(None, ))
embedded_question = embedding(question_input)

embedded_question1 = Conv1D(64, 5, activation='relu', strides=1)(embedded_question)
embedded_question1 = SpatialDropout1D(0.2)(embedded_question1)
embedded_question1 = GlobalMaxPooling1D()(embedded_question1)
embedded_question1 = Flatten()(embedded_question1)

embedded_question2 = Conv1D(128, 5, activation='relu', strides=1)(embedded_question)
embedded_question2 = SpatialDropout1D(0.2)(embedded_question2)
embedded_question2 = GlobalMaxPooling1D()(embedded_question2)
embedded_question2 = Flatten()(embedded_question2)

embedded_question3 = Conv1D(256, 5, activation='relu', strides=1)(embedded_question)
embedded_question3 = SpatialDropout1D(0.2)(embedded_question3)
embedded_question3 = GlobalMaxPooling1D()(embedded_question3)
embedded_question3 = Flatten()(embedded_question3)

### 답변 layer
answer_input = Input(shape=(None, ))
embedded_answer = embedding(answer_input)

embedded_answer1 = Conv1D(64, 5, activation='relu', strides=1)(embedded_answer)
embedded_answer1 = SpatialDropout1D(0.5)(embedded_answer1)
embedded_answer1 = GlobalMaxPooling1D()(embedded_answer1)
embedded_answer1 = Flatten()(embedded_answer1)

embedded_answer2 = Conv1D(512, 5, activation='relu', strides=1)(embedded_answer)
embedded_answer2 = SpatialDropout1D(0.5)(embedded_answer2)
embedded_answer2 = GlobalMaxPooling1D()(embedded_answer2)
embedded_answer2 = Flatten()(embedded_answer2)

embedded_answer3 = Conv1D(256, 5, activation='relu', strides=1)(embedded_answer)
embedded_answer3 = SpatialDropout1D(0.5)(embedded_answer3)
embedded_answer3 = GlobalMaxPooling1D()(embedded_answer3)
embedded_answer3 = Flatten()(embedded_answer3)


title_concatenated = concatenate([embedded_title1, embedded_title2, embedded_title3])
question_concatenated = concatenate([embedded_question1, embedded_question2, embedded_question3])
answer_concatenated = concatenate([embedded_answer1, embedded_answer2, embedded_answer3])

## 제목과 질문 merge
tq_concatenated = concatenate([title_concatenated, question_concatenated])

## 내용과 답변 merge
ta_concatenated = concatenate([question_concatenated, answer_concatenated])

### category layer
category_input = Input(shape=(5,))
category = Dense(100, activation='relu')(category_input)
category = Dropout(0.2)(category)
category = Dense(10, activation='relu')(category)


### 질문 hidden layer
concatenated1 = concatenate([tq_concatenated, category])
output1 = Dense(100, activation='relu', kernel_regularizer=l2(0.001))(concatenated1)
output1 = Dropout(0.2)(output1)
output1 = Dense(21, activation='sigmoid')(output1)

### 답변 hidden layer
concatenated2 = concatenate([ta_concatenated, category])
output2 = Dense(100, activation='relu', kernel_regularizer=l2(0.001))(concatenated2)
output2 = Dropout(0.2)(output2)
output2 = Dense(9, activation='sigmoid')(output2)

optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model1 = Model([title_input, question_input, category_input], output1)
model1.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])
model1.summary()

model2 = Model([question_input, answer_input, category_input], output2)
model2.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['acc'])
model2.summary()

question_target = np.array(question_output)
answer_target = np.array(answer_output)

early_stopping = EarlyStopping(monitor='val_acc', patience = 5)

history1 = model1.fit([train_title_pad, train_body_pad, encode_category],
                      question_target,
                      epochs=50, validation_split=0.3, batch_size=64)

### Test Data 전처리 및 학습

test_title = tokenizer.texts_to_sequences(test_data.question_title)
test_question = tokenizer.texts_to_sequences(test_data.question_body)
test_answer = tokenizer.texts_to_sequences(test_data.answer)
test_title_pad = pad_sequences(test_title)
test_question_pad = pad_sequences(test_question)
test_answer_pad = pad_sequences(test_answer)


test_categoria = test_data.category

## category 데이터를 라벨 숫자로 변환
test_categoria_pad = le.fit_transform(test_categoria)
## 변환된 라벨을 원핫 인코딩
encode_test_category = tf.keras.utils.to_categorical(test_categoria_pad)

history2 = model2.fit([train_body_pad, train_answer_pad, encode_category],
                      answer_target,
                      epochs=50, validation_split=0.3, batch_size=64)

test_target_q = model1.predict([test_title_pad, test_question_pad, encode_test_category], batch_size=64)
test_target_a = model2.predict([test_question_pad, test_answer_pad, encode_test_category], batch_size=64)

test_target = np.concatenate((test_target_q,test_target_a),axis=1)

sample_submission = []
for i in range(len(np.array(test_data.qa_id).tolist())):
    n = test_target.tolist()[i]

    n.insert(0,np.array(test_data.qa_id).tolist()[i])
    sample_submission.append(n)

submission = pd.DataFrame(sample_submission)


yy = train_data.columns[11:]
headers = list(yy)
headers.insert(0,"qa_id")
submission.columns = headers
submission.to_csv('submission.csv', index=False)
submission.head()