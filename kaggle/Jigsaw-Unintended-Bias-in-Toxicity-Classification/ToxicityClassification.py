import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional,Dropout

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

print('?')

train=pd.read_csv('train.csv',encoding='utf-8')
len(train)
test= pd.read_csv('test.csv', encoding='utf-8')
len(test)

train.info()

x_train = train['comment_text'].astype(str)
y_train = train['target']
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = test['comment_text'].astype(str)

check=train.head(300)

########EDA
train['target'] = np.where(train['target'] >= 0.5, True, False)
import matplotlib.pyplot as plt
plt.figure()
train['target'].value_counts().plot(kind='bar')
plt.title('Toxic vs Non Toxic Comments')
plt.show()

aaa=train[train.target==1]['comment_text'].iloc[:100]

print(train[train.severe_toxicity==1]['comment_text'].iloc[0])
print(train[train.obscene==1].iloc[1,2])
train['comment_length'] = train['comment_text'].apply(lambda x : len(x))
plt.figure(figsize = (12, 5))
plt.title('comment_length < 200')
plt.hist(train['comment_length'], bins = 60, range = [0, 200], alpha = 0.5, color = 'b')
plt.show()

train_length = train.comment_text.apply(lambda x : len(x.split()))
train_length.max()

plt.figure(figsize = (12, 5))
plt.title('word length')
plt.hist(train_length, bins = 60, range = [0, 320], alpha = 0.5, color = 'b')
plt.show()

########데이터 전처리

#미스 스펠 사전 정의
mispell_dict = {'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times',
                'garycrum':'gary crum','htmlutmterm':'html utm term','RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned',
                'Vinis':'vinys','Yᴏᴜ':'you','Trumpsters':'trump','Trumpian':'trump','bigly':'big league','Trumpism':'trump','Yoyou':'you','Auwe':'wonder','Drumpf':'trump',
                'utmterm':'utm term','Brexit':'british exit','utilitas':'utilities','ᴀ':'a', '😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian',
                'deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine',
                'fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations',
                'dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists'}

mispell_dict1 = {'whattsup': 'WhatsApp', 'whatasapp':'WhatsApp', 'whatsupp':'WhatsApp','whatcus':'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat':'what',
                 'Whwhat': 'What', 'whatshapp':'WhatsApp', 'howhat':'how that','Whybis':'Why is', 'laowhy86':'Foreigners who do not respect China','Whyco-education':'Why co-education',
                 "Howddo":"How do", 'Howeber':'However', 'Showh':'Show',"Willowmagic":'Willow magic', 'WillsEye':'Will Eye', 'Williby':'will by','pretextt':'pre text',
                 'aɴᴅ':'and','amette':'annette','aᴛ':'at','Tridentinus':'mushroom','dailycaller':'daily caller', "™":'trade mark'}

mispell_dict.update(mispell_dict1)

#욕설 사전 정의
swear_dict = {'c0ck' : 'cocksucker', 'c0cksucker': 'cocksucker', 'cock_sucker':'cocksucker', 'cl1t':'clit', 'd1ck':'dick', 'dog-fucker': 'doggin', 'f4nny':'fuck','fux0r':'fuck','f_u_c_k':
              'fuck', 'god-dam':'God', 'god-damned':'God', 'jack-off':'jackoff','jerck-off':'jarckoff','d1ck':'dick','b!tch':'bitch','b17ch':'bitch','b1tch':'bitch','v14gra':'viagra',
              'v1gra':'viagra','vagina':'viagra','p0rn':'porn','n1gga':'nigger','n1gger':'nigger','nigg3r':'nigger','nigg4h':'nigger','mo-fo':'mothafucker','mof0':'mothafucker','tw4t':'twat',
              '13i+ch':'bitch','13itch':'bitch','m45terbate':'masterbate','ma5terb8':'masterbate','ma5terbate':'masterbate','master-bate':'masterbate','masterb8':'masterbate','masterbat*':'masterbate',
              'masterbat3':'masterbate','sh!+':'shit','sh!t':'shit','sh1t':'shit','shi+':'shit','son-of-a-bitch':'bitch','s_h_i_t':'shit','t1tt1e5':'titt','t1tties':'titt','tittie5':'titt','w00se':'wang',
              'l3i+ch':'bitch','l3itch':'bitch','5h1t':'shit','5hit':'shit'}


def correct_spelling(x, dic):
    for word in dic.keys():
        if word in x:
            x = x.replace(word, dic[word])
    return x

x_train = x_train.apply(lambda x: correct_spelling(x, mispell_dict))
x_test = x_test.apply(lambda x: correct_spelling(x, mispell_dict))

def trans_swear(x, dic):
    for word in dic.keys():
        if word in x:
            x = x.replace(word, dic[word])
    return x

x_train = x_train.apply(lambda x: trans_swear(x, swear_dict))
x_test = x_test.apply(lambda x: trans_swear(x, swear_dict))

#불용어 정의
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
              "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
              "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
              "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
              "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
              "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that",
              "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
              "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
              "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)


remove_stopwords_train = x_train.apply(remove_stopwords)
remove_stopwords_test = x_test.apply(remove_stopwords)

#특수문자 정의
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '\n', '\r']


def clean_punct(x) :
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' {} '.format(punct))
    return x

x_train = x_train.apply(lambda x : clean_punct(x))
x_test = x_test.apply(lambda x : clean_punct(x))

#숫자제거
def clean_numbers(x):
    return re.sub('\d+', ' ', x)

x_train=x_train.apply(lambda x : clean_numbers(x))
x_test=x_test.apply(lambda x : clean_numbers(x))

embedding_dim = 300
max_length = 220


CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

tokenizer = Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train) + list(x_test))

#train x test x train y 정의
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_sequences = tokenizer.texts_to_sequences(x_train) #각 단어마다 정수값을 줘서 문장을 벡터로 변환
train_padded = pad_sequences(train_sequences,maxlen=max_length) #가장 길이가 긴 벡터 기준으로 문장 벡터의 길이를 같게 해줌

test_sequences = tokenizer.texts_to_sequences(x_test)
test_padded = pad_sequences(test_sequences ,maxlen=max_length)

Y_train = np.where(y_train >= 0.5, True, False)


#임베딩 매트릭스 생성
EMBEDDING_FILES = [
    'crawl-300d-2M.vec/crawl-300d-2M.vec',
    'glove.840B.300d.txt/glove.840B.300d.txt'
]

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix

embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
print('made matrix')

#sample weight 추가
train=pd.read_csv('train.csv',encoding='utf-8')
id_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
target_column = 'target'

for column in id_columns + [target_column]:
    train[column] = np.where(train[column] >= 0.7, True, False)

sample_weights = np.ones(len(train_padded), dtype=np.float32)
sample_weights += train[id_columns].sum(axis=1)

sample_weights += train[target_column] * (~train[id_columns]).sum(axis=1) #target은 독소가 있다고 나오는데, id_columns 정보가 없으면 가중
sample_weights += (~train[target_column]) * train[id_columns].sum(axis=1)  #target은 독소가 없다고 나오는데, id_columns 정보가 있으면 가중 <-굳이..?
sample_weights += (~train[target_column] & train['homosexual_gay_or_lesbian'] + 0) *5
# sample_weights += (~train[target_column] & train['black'] + 0) * 5
# sample_weights += (~train[target_column] & train['white'] + 0) * 5
# sample_weights += (~train[target_column] & train['muslim'] + 0) * 1
# sample_weights += (~train[target_column] & train['jewish'] + 0) * 1
sample_weights /= sample_weights.mean()

#모델링
def build_model(embedding_matrix):
    post_input= Input(shape=(max_length,))
    x=Embedding(embedding_matrix.shape[0],embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(post_input)
    x=SpatialDropout1D(0.2)(x)
    x=Bidirectional(LSTM(128, return_sequences=True))(x)
    x=Bidirectional(LSTM(128))(x)
    # x=Dropout(0.2)(x)
    x= Dense(512,activation='relu')(x)
    x=Dropout(0.1)(x)
    x= Dense(512,activation='relu')(x)
    target_prediction= Dense(1, activation='sigmoid',name='target')(x)
    aux_prediction= Dense(y_aux_train.shape[-1],activation='sigmoid',name='aux')(x)
    model= Model(inputs=post_input, outputs=[target_prediction,aux_prediction])
    model.compile(loss='binary_crossentropy', optimizer='nadam',loss_weights=[1, 2.8])

    return model


model = build_model(embedding_matrix)
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
from tensorflow.keras.callbacks import LearningRateScheduler

checkpoint_predictions = []
weights = []

epoch=4

for i in range(2):
    model = build_model(embedding_matrix)
    for ep in range(epoch):
        model.fit(train_padded, [Y_train, y_aux_train],
                  batch_size=256,
                  epochs=1,
                  verbose=1,
                  sample_weight=[sample_weights,np.ones_like(sample_weights)],
                  callbacks=[LearningRateScheduler(lambda _: 1e-3 * (0.55 ** ep))
                             ]
                  )
        checkpoint_predictions.append(model.predict(test_padded,batch_size=1024)[0].flatten())
        weights.append(2 ** (ep*1.2))

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission=pd.read_csv('sample_submission.csv')
submission['prediction']=predictions
submission.head(10)
submission.to_csv('jigsaw_submission(67).csv', index=False)
