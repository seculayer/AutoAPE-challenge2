import tensorflow as tf
import numpy as np

from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm

import preprocess


# bert 모델의 마지막 hidden layer 에서 임베딩 벡터 가져옴
# 해당 임베딩 벡터에서, A/B/Pronoun 에 해당되는 애들만 가져옴
def run_bert(data):
    text = data["Text"]

    tokens = []
    token_ids = []
    embeddings = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')

    for txt in tqdm(text[:]):
        token = tokenizer.tokenize(txt)
        input_ids = tf.constant(tokenizer.encode(txt))[None, :]
        outputs = model(input_ids)
        last_hidden_states = outputs[0]

        tokens.append(token)
        token_ids.append(input_ids)
        embeddings.append(last_hidden_states)

        #  (n, 1, 75,768)  / emb, emb[0], emb[0][0], emb[0][0][0]
    print(f' embedding check : {len(embeddings)} ')
    print(f' embedding check : {len(embeddings[0])} ')
    print(f' embedding check : {len(embeddings[0][0])} ')
    print(f' embedding check : {len(embeddings[0][0][0])} ')

    emb_list = []

    for i in range(len(data)) :

        my_token = tokens[i]
        my_emb = embeddings[i][0]

        P = data.loc[i, "Pronoun"].lower()
        A = data.loc[i, "A"].lower()
        B = data.loc[i, "B"].lower()

        P_offset = preprocess.no_space_offset(data.loc[i, "Text"], data.loc[i, "Pronoun-offset"])
        A_offset = preprocess.no_space_offset(data.loc[i, "Text"], data.loc[i, "A-offset"])
        B_offset = preprocess.no_space_offset(data.loc[i, "Text"], data.loc[i, "B-offset"])

        A_length = preprocess.rmv_special_chars(A)
        B_length = preprocess.rmv_special_chars(B)

        P_length = preprocess.rmv_special_chars(P)

        emb_A = np.zeros(768)
        emb_B = np.zeros(768)
        emb_P = np.zeros(768)

        count_chars = 0
        cnt_A, cnt_B, cnt_P = 0, 0, 0

        for j in range(len(my_token)):
            tks = my_token[j]

            # if count_chars == P_offset:
            # if count_chars in range(P_offset, P_offset + P_length):
            if count_chars in range(P_offset, P_offset + P_length):

                # print(f'토큰 : {tks}')
                emb_P += np.array(my_emb[j])
                cnt_P += 1

            # elif count_chars != P_offset :
            #   print(f' {count_chars} 없음 p offset : {P_offset}')

            if count_chars in range(A_offset, A_offset + A_length):
                emb_A += np.array(my_emb[j])
                cnt_A += 1

            if count_chars in range(B_offset, B_offset+B_length):
                emb_B += np.array(my_emb[j])
                cnt_B += 1

            count_chars += preprocess.rmv_special_chars(tks)

        emb_A /= cnt_A
        emb_B /= cnt_B

        my_emb = []
        emb_A = emb_A.tolist()
        emb_B = emb_B.tolist()
        emb_P = emb_P.tolist()
        my_emb.append(emb_A)
        my_emb.append(emb_B)
        my_emb.append(emb_P)

        emb_list.append(my_emb)

    return emb_list