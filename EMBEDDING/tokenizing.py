from keras.preprocessing.text import Tokenizer
import json

# 전체 입력 문자열에 대해서 Token화를 진행
tokenizer = Tokenizer()
tokenizer.fit_on_texts(final['BZ_PPOS_ITM_CTT'])

train_seq = tokenizer.texts_to_sequences(final['BZ_PPOS_ITM_CTT'])
word_vocab = tokenizer.word_index
MAX_SEQ_LENGTH = 50 # 각 문장의 최대 길이


print('전체 단어의 개수' , len(word_vocab))
# 전체 단어의 개수 1130685

# 평균 단어의 길이가 50개정도라서 MAXLEN을 50으로 설정
# 대분류 학습 데이터 -> np.array로 저장
from keras.preprocessing.sequence import pad_sequences
train_inputs = pad_sequences(train_seq, maxlen = MAX_SEQ_LENGTH, padding = 'post')

# 대분류 코드값 object -> int
# 대분류 코드 학습에 쓰일 input label target data -> np.array로 저장
from sklearn.preprocessing import LabelEncoder
import numpy as np
encoder = LabelEncoder()
encoder.fit(np.array(final['Large']))

train_labels = np.array(encoder.transform(final['Large'])) 

# 단어 사전 만들기
data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)+1



# DATA_PATH = '/content/drive/MyDrive/KEDxBOAZMY/DATA/ModelTrainResults'
# TRAIN_INPUT = 'train_input.npy'
# TRAIN_LABEL = 'train_label.npy'
# DATA_CONFIGS = 'data_configs.json'