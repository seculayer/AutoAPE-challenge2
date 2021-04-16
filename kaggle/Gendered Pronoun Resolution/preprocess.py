import re
import math
import tensorflow as tf

# 띄어쓰기 제외한 텍스트 길이
def no_space_offset(text, offest):
    cnt = 0
    for i in range(offest):
        if text[i] not in " ":
            cnt += 1
    return cnt


# 특수문자 제외한 텍스트 길이
def rmv_special_chars(text):
    cnt = 0
    for i in range(len(text)):
        if text[i] not in "#" :
            cnt += 1
    return cnt


# 띄어쓰기 + 특수문자 제외한 텍스트 길이
def no_spc_spec_lenght(text):
    cnt = 0
    my_txt = str(text)

    for i in range(len(text)):
        if my_txt[i] not in "#" and my_txt[i] not in " ":
            cnt += 1
    return cnt


# 레이블 생성
def get_labels (data):
    labels = []

    for i in range(len(data)) :
        label = [0, 0, 1]
        if (data.loc[i, "A-coref"] == True):
            label = [1, 0, 0]
        if (data.loc[i, "B-coref"] == True):
            label = [0, 1, 0]
        labels.append(label)

    return labels


# 문장 => 단어별로 split
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


# 단어 토큰화
def get_train_set (tokens, lables):

    tokens_with_len = [ [words, lables[i], len(words)]
                        for i, words in enumerate(tokens)]

    # 길이로 sort
    tokens_with_len.sort(key=lambda x: x[2])

    # 길이 속성 제거
    sorted_tokens_lables = [(ele[0], ele[1]) for ele in tokens_with_len]

    processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_tokens_lables, output_types=(tf.int32, tf.int32))
    BATCH_SIZE = 2
    batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))

    TOTAL_BATCHES = math.ceil(len(sorted_tokens_lables) / BATCH_SIZE)
    TEST_BATCHES = TOTAL_BATCHES // 10

    final_dataset = []
    train_data = batched_dataset.skip(TEST_BATCHES)
    test_data = batched_dataset.take(TEST_BATCHES)
    final_dataset.append(train_data)
    final_dataset.append(test_data)

    return final_dataset










