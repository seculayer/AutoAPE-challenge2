import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D,GRU
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np


if __name__ == "__main__":
    train=pd.read_csv("../188_quora/data/train.csv")
    test=pd.read_csv("../188_quora/data/test.csv")
    train = train.loc[1:1000]

    # Convert to lower case
    train, val = train_test_split(train, test_size=0.1, random_state=2019)
    tokenizer = Tokenizer(num_words=10000,  # how many unique words to use
                          filters=''  # characters that will be filtered from the texts
                          )
    tokenizer.fit_on_texts(list(train.question_text)) #+ list(test.question_text) + list(val.question_text))
    X = tokenizer.texts_to_sequences(train.question_text)
    X_test = tokenizer.texts_to_sequences(test.question_text)
    vocabulary = tokenizer.word_index
    print('tokenize done')
    X_embed = pad_sequences(X,
                            maxlen=100,  # Max. number of words in sentence
                            padding='pre'  # Where to add padding
                            )
    X_test_embed = pad_sequences(X_test,
                                 maxlen=100,  # Max. number of words in sentence
                                 padding='pre'  # Where to add padding
                                 )

    inp = Input(shape=(100,))
    x = Embedding(10000, 300)(inp)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ty2 = val['target'].values
    ty = train['target'].values
    model.fit(X_embed, ty, batch_size=512, epochs=1)

    pred_y = model.predict([X_test_embed], batch_size=1024, verbose=1)
    pred_test_y = pred_y
    print('done')
    temp = []
    for i in range(len(test['qid'])):
        temp.append(0)
    a = np.array(temp)
    fin = np.reshape(a, (-1, 1))



    out_df = pd.DataFrame({"qid": test["qid"].values})

    out_df['prediction'] = fin
    out_df.to_csv("submission.csv", index=False)
