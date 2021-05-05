def remove_stopwords(data, path):
    import numpy as np
    import pandas as pd
    """
    data -> DataFrame 형태로 입력
    path ->  stopwords의 저장경로
    """
    stopwords = pd.read_excel(path)
    stopwords = np.array(stopwords['불용어'])
    result = []
    """
    stopwords는 불용어들을 담은 리스트를 입력
    data는 역시나 dataframe의 형태로 입력될 것이며
    모든 행에 대해서 data['BZ_PPOS_ITM_CTT']의 단어들 중에서 stopword에 포함되어 있으면 제거
    
    최종적으로 불용어를 제거한 문자열들을 담은 result 리스트를 return
    """
    for seq in data['BZ_PPOS_ITM_CTT'].astype(str):
        seq = seq.split(' ')
        curr = ''
        for word in seq:
            if word in stopwords:
                continue
            else:
                curr += ' ' +word
        result.append(curr.strip())
    
    return result