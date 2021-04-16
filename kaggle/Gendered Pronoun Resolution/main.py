import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
import pandas as pd
import models, preprocess, mybert

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping


# 데이터셋 로드 => bert로 전처리 => label 생성
test_data = pd.read_csv("gap-test.tsv", sep = '\t')
test_emb = mybert.run_bert(test_data[:])
test_labels = preprocess.get_labels(test_data[:])

validation_data = pd.read_csv("gap-validation.tsv", sep = '\t')
val_emb = mybert.run_bert(validation_data[:])
val_labels = preprocess.get_labels(validation_data[:])

development_data = pd.read_csv("gap-development.tsv", sep = '\t')
dev_emb = mybert.run_bert(development_data)
dev_labels = preprocess.get_labels(development_data[:])


submission_data = pd.read_csv("test_stage_2.tsv", sep = '\t')
sub_emb = mybert.run_bert(submission_data)


x_test = np.array(test_emb)
y_test = np.array(test_labels)

x_dev = np.array(dev_emb)
y_dev = np.array(dev_labels)

x_val = np.array(val_emb)
y_val = np.array(val_labels)

x_sub = np.array(sub_emb)


# 모델 정의 & 학습
input_shape = x_dev.shape[1:]
# y_dev = y_dev.reshape(len(y_dev), 1, y_dev.shape[1])
model = models.my_lstm(input_shape, 3)

filename = 'checkpoint-epoch-{}-trial-001.h5'.format(25)
early_stopping = EarlyStopping()
checkpoint = ModelCheckpoint(filename,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto'
                             )

model.fit(x_dev, y_dev, epochs=25, verbose=1, callbacks=[checkpoint, early_stopping])

test_data = pd.read_csv('./test_stage_2.csv')
test_text = test_data["Text"]
test_df = pd.DataFrame(test_text)


#  테스트 데이터셋 predict & submission 파일 생성
result = model.predict(x_sub, verbose=1)

result_df = pd.read_csv('./sample_submission_stage_2.csv')

result_df.loc[:, "A"] = pd.Series(result[:, 0])
result_df.loc[:, "B"] = pd.Series(result[:, 1])
result_df.loc[:, "NEITHER"] = pd.Series(result[:, 2])

result_df.to_csv("submission_result.csv", index=False)






























# ---------------------------------------------------------------------------------------------------------
# data = pd.read_csv('./dataset/gap-validation.csv')
# text = data["Text"]
# label = data[["A-coref", "B-coref"]]

# text_df = pd.DataFrame(text)
# label_df = pd.DataFrame(label)

# dev_data = pd.read_csv('./dev_embeddings.csv')
# val_data = pd.read_csv('./val_embeddings.csv')
test_data = pd.read_csv('./test_embeddings.csv')

# emb_A, emb_B, emb_P, label
# dev_train = preprocess.get_train(dev_data)
# val_train = preprocess.get_train(val_data)
test_train = preprocess.get_train(test_data)

# dev_labels = preprocess.get_lables(dev_data)
# val_labels = preprocess.get_lables(val_data)
test_labels = preprocess.get_lables(test_data)


input_shape = (10, 153)
model = models.my_cnn(input_shape, 3)

model.fit(test_train, test_labels, epochs=50, verbose=1)


#  전처리
test_data = pd.read_csv("gap-test.tsv", sep = '\t')
test_emb = mybert.run_bert(test_data[:])
test_labels = preprocess.get_labels(test_data[:])


validation_data = pd.read_csv("gap-validation.tsv", sep = '\t')
val_emb = mybert.run_bert(validation_data[:])
val_labels = preprocess.get_labels(validation_data[:])

development_data = pd.read_csv("gap-development.tsv", sep = '\t')
dev_emb, dev_labels = mybert.run_bert(development_data)
dev_labels = preprocess.get_labels(development_data[:])


submission_data = pd.read_csv("test_stage_2.tsv", sep = '\t')
sub_emb = mybert.run_bert(submission_data)


#  emb : [개수][원소 리스트][원소]
# emb_A, emb_B, emb_P, label

x_test = np.array(test_emb)
y_test = np.array(test_labels)

x_dev = np.array(dev_emb)
y_dev = np.array(dev_labels)

x_val = np.array(val_emb)
y_val = np.array(val_labels)

x_sub = np.array(sub_emb)


input_shape = x_test.shape[1:]
model = models.my_cnn(input_shape, 3)

print(f'shape : {x_test.shape}, {y_test.shape}')


model.fit(x_test, y_test, epochs=50, verbose=1)

test_data = pd.read_csv('./test_stage_2.csv')
test_text = test_data["Text"]
test_df = pd.DataFrame(test_text)


# sub_data
sub_data = x_test

result = model.predict(sub_data, verbose=1)

result_df = pd.read_csv('./sample_submission_stage_2.csv')
result_df.loc[:, "A"] = pd.Series(result[:, 0])
result_df.loc[:, "B"] = pd.Series(result[:, 1])
result_df.loc[:, "NEITHER"] = pd.Series(result[:, 2])


result_df.to_csv("submission.csv", index=False)




