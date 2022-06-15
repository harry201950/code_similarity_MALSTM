

# problem 파일 경로 Desktop/project/code_similarity/open/code

# problem 폴더 안 예제 코드 확장자명 변경
import os
import glob

# .txt 병합
import pandas as pd
import numpy as np
import sklearn

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import scipy


# 폴더 및 .py 파일 개수 확인
path = "./open/code/"
dataset = os.listdir(path)

# ds.store숨김
dataset = [folder for folder in dataset if folder.startswith('problem')]

# 폴더명 정렬
dataset.sort(reverse=False)
len(dataset)


# # code - code/problem/.py 모두 불러옴
# cnt = 0
# for i in range(len(dataset)):
#     print(f"{dataset[i]} \t : \t {len(os.listdir(path + dataset[i]))}")
#     cnt += len(os.listdir(path + dataset[i]))


# .py 읽어오기
# file = glob.glob(path + "*/*.py")

# 파일명 .txt 변환
# for name_ext in file:
#     if not os.path.isdir(name_ext):
#         src = os.path.splitext(name_ext)
#         os.rename(name_ext, src[0]+'.txt')


# txt 파일 DF 저장
csv_tr = []
csv_tr2 = []
similarity = []

for i in glob.glob("./open/code/*/*.txt"):
    with open(os.path.join(os.getcwd(), i), 'r') as f:
        text = f.read()
        #무작위 행 셔플
        csv_tr.append(text)
        csv_tr2.append(text)
        similarity.append(-1)
# 무작위 행 셔플
csv_tr = sklearn.utils.shuffle(csv_tr)

# DF 생성
code_csv = pd.DataFrame(zip(csv_tr, csv_tr2,similarity),
                        columns=['code1', 'code2','similarity'])
#code_csv.to_csv("code_csv.csv")

# code_csv.to_csv("code_csv.csv")


# %% define Model

# TF-idf행렬의 크기
tfidV = TfidfVectorizer(max_features=10000)
tfidV_matrix1 = tfidV.fit_transform(code_csv['code1'])  # fit(train['code1'])
tfidV_matrix2 = tfidV.fit_transform(code_csv['code2'])

print("행렬의 크기: ", tfidV_matrix1.shape)
print("행렬의 크기: ", tfidV_matrix2.shape)

tfidV_matrix1.toarray()
tfidV_matrix2.toarray()

#similarity유사도 구하기
cosine_sim = cosine_similarity(tfidV_matrix1, tfidV_matrix2) 

# df추가
code_csv['similarity'] = cosine_sim

#값 변환
threshold = 0.5
code_csv['preds'] = np.where(code_csv['cosine_sim'] > threshold, 1, 0)

code_csv

# %%
# # Define Model (CountVectorizer+CosineSimilarity)
# class BaselineModel():
#     def __init__(self, threshold=0.5):
#         super(BaselineModel, self).__init__()
#         self.threshold = threshold  # 유사도 임계값
#         self.TfidfVectorizer = TfidfVectorizer()

#     def fit(self, code1, code2):
#         # 입력 받은 코드 쌍으로 부터 vectorizer를 vector화
#         code1 = self.TfidfVectorizer.fit_transform(code1)
#         code2 = self.TfidfVectorizer.fit_transform(code2)
#         print('Done1.')

#     def predict_proba(self, code1, code2):

#         # 입력 받은 코드 쌍으로 부터 vectorizer를 vector화
#         code1_vecs = self.TfidfVectorizer.fit_transform(code1)
#         code2_vecs = self.TfidfVectorizer.fit_transform(code2)

#         preds = []

#         # 각각의 코드 쌍(=벡터 쌍)으로부터 cosine-similarity를 구합니다.
#         for code1_vec, code2_vec in zip(code1_vecs, code2_vecs):
#             preds.append(cosine_similarity(code1_vec, code2_vec))

#         # preds = np.reshape(preds, len(preds))
#         preds = np.reshape(preds, len(preds))

#         print('Done2.')

#         # 각 코드 쌍들의 유사도를 반환
#         return preds

#     def predict(self, code1, code2):
#         preds = self.predict_proba(code1, code2)

#         # cosine-similarity (유사도)가 설정한 임계값(Threshold=0.5)보다 높다면 유사하다 : 1, 아니라면 유사하지 않다 : 0
#         preds = np.where(preds > self.threshold, 1, 0)

#         # 각 코드 쌍들의 유사도를 Threshold를 통해 유사함을 판별 (이진분류)
#         return preds


# # Model(Vectorizer) Fit
#%%

# 모델 선언
model = BaselineModel(threshold=0.5)

# 학습 코드 쌍들로부터 Model을 Fitting
model_csv = model.fit(train['code1'], train['code2'])


# Inference
# 모델 추론
preds = model.predict(test['code1'], test['code2'])
# Submission


submission = pd.read_csv('./open/sample_submission.csv')
submission['similar'] = preds
submission.to_csv('./submission.csv', index=False)
