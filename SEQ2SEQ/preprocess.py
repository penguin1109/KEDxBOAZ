def str_to_words(df):
  for str in df:
    text = str[4:-4].split("', '")
    words.append(text)

str_to_words(df['BZ_PPOS_ITM_CTT'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(words)
text_sequences = tokenizer.texts_to_sequences(words)

word_dic = tokenizer.word_index
word_dic['<PAD>'] = 0
# 총 1142004개의 단어로 이루어진 단어 사전

# 최대 길이 77로 벡터화를 진행
train_inputs = pad_sequences(text_sequences, maxlen = 77, padding = 'post')

