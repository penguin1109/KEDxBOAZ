def data_tokenizing(data, length, code):
    """
    data -> dataframe의 형태
    length -> int
    length 매개변수는 최대 사용할 문자열의 길이의 설정을 위해서이다.
    code 매개변수는 대/중/소/세/세세 분류 중에 어떤 것을 예측할지 설정해 주는 용도이다.
    """
    CODE = {'대분류' : 1, '중분류' : 2, '소분류' : 3, '세분류' : 4}

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['BZ_PPOS_ITM_CTT'].astype(str))
    
    train_seq = tokenizer.texts_to_sequences(data['BZ_PPOS_ITM_CTT'].astype(str))
    word_vocab = tokenizer.word_index
    
    MAX_SEQ_LENGTH = length # 사용할 문자열의 최대 길이
    
    train_inputs = pad_sequences(train_seq, maxlen = MAX_SEQ_LENGTH, padding = 'post')
    
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    label_size = CODE[code]
    train_labels = np.array(list(map(lambda x: str(x)[:label_size], dfB.index.values)))
    encoder.fit(train_labels)
    
    train_labels = np.array(encoder.transform(train_labels)) # label데이터에 encoding을 적용
    
    data_configs = {} # 입력한 데이터 문자열에서 추출한 단어로 만든 단어 사전
    data_configs['vocab'], data_configs['vocab_size'] = word_vocab, len(word_vocab)+1
    
    """
    train_inputs -> 토큰화가 진행되고 padding 또한 진행된 데이터를 반환
    -> 모델에 입력할 수 있는 형태
    train_labels -> 수치형 데이터로 예측해야하는 업종 코드를 반환
    data_configs -> 단어 사전과 단어의 총 개수를 dictionary의 형태로 입력된 데이터를 반환
    """
    return train_inputs, train_labels, data_configs