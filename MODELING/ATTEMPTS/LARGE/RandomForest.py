from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

forest = RandomForestClassifier(n_estimators = 120)
t_x, v_x, t_y, v_y = train_test_split(data, label, test_size = 0.25, random_state = 42)

forest.fit(t_x, t_y)

print('Accuracy : {}'.format(forest.score(v_x, v_y)))
Accuracy : 0.5914803594200335

data # Embedding을 시켜준 데이터bedding을 시켜준 데이터
array([[    161,     525,       2, ...,       0,       0,       0],
       [    283,      39,   25477, ...,       0,       0,       0],
       [   1392,       3,      18, ...,     161,     390,    2493],
       ...,
       [ 311646,       3,      17, ...,       0,       0,       0],
       [    568,    5418,    1310, ...,       0,       0,       0],
       [1130683, 1130684,  100098, ...,       0,       0,       0]])

label # 대분류 코드를 정수형 데이터로 바꾸어준 입력 target 데이터
array([6, 2, 6, ..., 6, 6, 2], dtype=int64)