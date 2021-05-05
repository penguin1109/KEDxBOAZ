def clean_text(data):
    words = []
    # 무조건 문자열로 바꾸어주는 astype(str)
    data = data.astype(str)
    import re
    FILTERS = "([~.,!?\"':;)(])" 
    
    # 최종적으로 특수 문자를 모두 제거
    
    CHANGE_FILTER = re.compile(FILTERS)
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split(): # 공백을 기준으로 단어들을 나눔
            words.append(word)
    return list(set(word for word in words if word)) # 전체 데이터의 모든 데이터를 포함하는 단어 리스트를 만듬
    # 반환해주는 단어 리스트의 경우에 list(set())이기 때문에 중복되는 단어는 제거 된 채로 반환