def test_to_csv(path):
    """
    최종 결과 예측을 하기 위한 입력 데이터를 처리하는 함수
    test.txt의 경로가 주어지면 이를 csv 파일로 바꾸어서 반환
    """
    import re
    import pandas as pd
    with open(path, encoding = 'utf-8') as f:
        result = dict()
        for line in f:
            line = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', line) 
      # 1. 모든 특수 문자 제거
      # 2. 회사 번호와 각회사에 따른 정보를 분리
      # 3. 정보에 대해서는 숫자를 모두 제거하고 앞뒤 공백또한 제거
      # 4. 처리된 데이터를 하나의 정보로 모아준 뒤에 사전형 데이터인 result에 저장 후 dataframe의 형태인 df로 바꾸어서 return
            code = line[1:8]
            info = re.sub('[0-9]+', '' , line[11:].strip())

            if code not in result:
                result[code] = info
            else:
                result[code] += ' ' + info
  
    df = pd.DataFrame(data = result.values(), index = result.keys(), columns = ['BZ_PPOS_ITM_CTT'])

    return df