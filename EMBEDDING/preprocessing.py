import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

PAD, STD, END, UNK = "<PAD>", "<STD>", "<END>", "<UNK>"
PAD_IDX, STD_IDX, END_IDX, UNK_IDX = 0, 1, 2, 3
Marker = [PAD, STD, END, UNK]
Pattern = "([~.,!?\"':;)(])"
Change = re.compile(pattern = Pattern)

# 정규표현식을 사용해서 특수 기호를 제거
# 공백 문자를 기준으로 단어들을 나누어서 데이터의 모든 단어를 포함하는 단어 리스트로 변형
def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(Change, "", sentence)
        for word in sentence.split():
            words.append(word)
    return[word for word in words if word]

# 한글 텍스트를 tokenizing하기 위해 형태소로 분리하고자 한다.
# 이는 KoNLPy의 Okt형태소 분리기를 사용하여 분리할 것이다.
def prepo_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = []
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ','')))
        result_data.append(morphlized_seq)
    return result_data

# word2idx라는 리스트를 key가 단어이고 value가 index인 단어 사전을 만든다.
# idx2word라는 리스트를 key가 index이고 value가 단어인 단어 사전을 만든다.
def make_vocab(vocab_list):
    word2idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx2word = {idx: word for idx, word in enumerate(vocab_list)}
    return word2idx, idx2word

# 인코더에 입력할 데이터를 만든다.
def encoder_input(value, dictionary, tokenize_as_morph = False):
    sequence_input_index, sequence_length = [], []
    if tokenize_as_morph:
        value = prepo_like_morphlized(value)
    for seq in value:
        seq = re.sub(Pattern, "''", seq)
        
def remove_split(df):
    df = df.replace("[", "")
    df = df.replace("]", "")
    df = df.split("', '")
    return df
df['BZ_PPOS_ITM_CTT'] = df['BZ_PPOS_ITM_CTT'].apply(lambda x:remove_split(x))
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
 
import re
 
def cleanText(df):
    text = re.sub('[-=+#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', df)
    text = text.replace("n", "")
    text = text.replace(",", "")
    text = word_tokenize(text)
 
    return text  

    






