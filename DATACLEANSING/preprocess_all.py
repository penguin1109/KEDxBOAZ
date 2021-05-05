from DATACLEANSING import clean_text
from DATACLEANSING import remove_stopwords

def preprocess_all(data, iscsv = True):
    """
    data를 csv형태, 혹은 array의 형태로 입력 받아서 cleansing을 진행한 후에 
    result라는 문자열이 담긴 array로 한꺼번에 변환
    미리 만들어둔 clean_text와 remove_stopwords 함수를 사용
    """

    if iscsv == True:
    # 특수 문자 제거
        data = clean_text(data)
    # 불용어 제거
        data = remove_stopwords(data, "/content/drive/MyDrive/KEDxBOAZ/stopwords.xlsx", iscsv = False)
    else:
        data = clean_text(data, iscsv = False)
        data = remove_stopwords(data, "/content/drive/MyDrive/KEDxBOAZ/stopwords.xlsx", iscsv = False)
    return data