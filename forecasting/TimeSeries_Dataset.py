import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class train_set(Dataset):
    def __init__(self, train_x, train_y):
        self.dataset = train_x
        self.label =train_y
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.dataset[index,:,:-1], self.dataset[index,:,-1], self.label[index,:]

if __name__ == "__main__":
    df=pd.read_csv('./energy/train.csv', encoding='cp949')

    category_column_name = 'num' #전력소의 종류를 알려주는 컬럼이름
    num_feature = 8 #input에 사용될 특징의 갯수
    label_index = 2 #레이블 컬럼의 컬럼상 위치 

    category = df[category_column_name].unique().tolist() # 전력소의 종류를 리스트로 반환
    num_category = len(category) # 전력소의 종류의 갯수를 구함

    input_window = 24*14 # 앞의 14일을 볼 것이다.
    output_window = 24 *7 # 뒤의 1주일을 예측할 것이다.
    shift = 0 # input 바로 다음시간을 예측할 것이다.
    stride = 1 # input은 멀리 뛰기 없이 한 칸씩 가며 데이터셋을 구성할 것이다.

    window_per_category=np.full((num_category), 0) # 카테고리 별 윈도우 갯수를 저장할 array 생성

    for n in range(num_category):
        total = df[df[category_column_name]==category[n]].shape[0] # 해당 전력소의 전체 길이를 구하고
        num_window_per_category = (total - input_window - output_window - shift)// stride + 1 # 공식에 따라 윈도우 갯수를 구하고
        window_per_category[n] = num_window_per_category # 이를 전력소별로 저장한다.
        
    total_window_num = np.sum(window_per_category) # 전력소별 데이터의 길이(윈도우의 숫자)는 총 합은 최종 데이터셋의 길이가 된다. 

    train_x = np.zeros((total_window_num, input_window, num_feature+1)) # +1을 해주는 이유는 임베딩에 따라 전력소의 종류를 넣어주기 위함
    train_y = np.zeros((total_window_num, output_window))
    # train shape = (96420 ,336, 9)
    # label shape = (96420, 168)

    count = 0
    for series in range(num_category):
        for i in range(window_per_category[series]):
            data_start = i * stride # 훈련 데이터셋의 시작점은 stride가 1이기에 0 ,1, 2, 3 이렇게 됨
            data_end = data_start + input_window # 훈련 데이터셋의 종료지점은 input 윈도우의 길이만큼임
            label_start = data_end + shift # label start는 shift가 0이기에 data가 끝나는 시점임
            label_end = label_start + output_window # label end는 label start에 output window 를 더한 시점임
            
            train_x[count, :, :-1] = df.loc[df[category_column_name]==category[series]].iloc[data_start:data_end, label_index:] # 순차적으로 담을 건데 전력소순서에 따라 담게 된다. 
            train_y[count, :] =  df.loc[df[category_column_name]==category[series]].iloc[label_start:label_end, label_index] 
            train_x[count, :, -1] = category[series] # train_x 의 마지막 특징은 카테고리 임베딩을 위해 어떤 전력소였음을 명시해 준다.
            count+=1 # 최종 count는 총 데이터의 길이 96420이 될것이다.
            
            
    dataset = train_set(train_x, train_y)
    dataloder = DataLoader(dataset, batch_size=32, shuffle=True)