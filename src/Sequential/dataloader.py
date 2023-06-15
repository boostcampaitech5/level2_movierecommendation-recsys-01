import os
import pandas as pd
import numpy as np
import torch
from typing import Tuple
from tqdm import tqdm
from torch.utils.data import Dataset


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(data_dir, 'train/train_ratings.csv')).drop('time', axis=1)
    sub_df = pd.read_csv(os.path.join(data_dir, 'eval/sample_submission.csv'))
    
    return train_df, sub_df


def process_data(train_df: pd.DataFrame, max_len: int, k: int, n_neg_samples: int) -> Tuple[dict, int, int, dict]:
    item_idx = train_df['item'].unique()
    user_idx = train_df['user'].unique()
    
    user2idx = {user:idx for idx,user in enumerate(user_idx)}
    item2idx = {item:idx+1 for idx,item in enumerate(item_idx)}
    idx2item = {idx+1:item for idx,item in enumerate(item_idx)}
    n_items = len(item2idx)
    n_users = len(user2idx)
    
    train_df['user'] = train_df['user'].map(user2idx)
    train_df['item'] = train_df['item'].map(item2idx)
    
    total = train_df.groupby('user')['item'].apply(np.array)
    train = list()
    valid = list()
    valid_cand = list()
    infer = list()
    infer_cand = list()
    for user_idx, user_total in enumerate(tqdm(total)):
        # 전체 negative
        total_neg = np.setdiff1d(np.arange(1, n_items+1), user_total)
        # 제일 뒤 절반, 앞에서 랜덤 절반 추출
        valid_pos = np.random.choice(user_total[:-(k//2)], (k//2), replace=False)
        valid_pos = np.append(valid_pos, user_total[-(k-k//2):])
        # 전체 negative 중에 랜덤 추출
        valid_neg = np.random.choice(total_neg, n_neg_samples, replace=False)
        valid_pos_neg = np.append(valid_pos, valid_neg)
        valid_cand.append(valid_pos_neg)
        
        # valid_pos 제외
        total_train = user_total[~np.isin(user_total, valid_pos)]
        # train
        train_sample_idx = np.sort(np.random.choice(
            np.arange(0, total_train.size), min(total_train.size, max_len), replace=False))
        train_sample = total_train[train_sample_idx]
        train.append(train_sample)
        # valid
        valid_sample_idx = np.sort(np.random.choice(
            np.arange(0, total_train.size), min(total_train.size, max_len-1), replace=False))
        valid_sample = np.append(total_train[valid_sample_idx], n_items+1)
        if valid_sample.size < max_len:
            pad_len = max_len - valid_sample.size
            valid_sample = np.append([0]*pad_len, valid_sample)
        valid.append(torch.tensor(valid_sample).unsqueeze(0))
        # infer
        infer_sample_idx = np.sort(np.random.choice(
            np.arange(0, user_total.size), min(user_total.size, max_len-1), replace=False))
        infer_sample = np.append(user_total[infer_sample_idx], n_items+1)
        if infer_sample.size < max_len:
            pad_len = max_len - infer_sample.size
            infer_sample = np.append([0]*pad_len, infer_sample)
        infer.append(torch.tensor(infer_sample).unsqueeze(0))
        infer_cand.append(total_neg)
        
        # if user_idx == 20:
        #     print("user_total")
        #     print(user_total)
        #     print("total_train")
        #     print(total_train)
        #     print("valid_pos")
        #     print(valid_pos)
        #     print("train_sample_idx")
        #     print(train_sample_idx)
        #     print("train_sample")
        #     print(train_sample)
        #     return
        # print(valid_pos)
        # print(valid_pos_neg)
        # print(train_sample)
        # print(np.append(train_sample[1:], n_items+1))
        # print(infer_sample)
        # print(infer_sample.size)
        # return
    
    data = {'train': train,
            'valid': valid,
            'valid_cand': valid_cand,
            'infer': infer,
            'infer_cand': infer_cand}
    
    return data, n_users, n_items, idx2item


class BERT4RecDataset(Dataset):
    def __init__(self,
                 train_data: pd.Series,
                 n_users: int,
                 n_items: int,
                 max_len: int,
                 mask_prob: float):
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_idx: int) -> Tuple[torch.tensor, torch.tensor]: 
        seq = self.train_data[user_idx]
        masked_seq = []
        labels = []
        for item_idx in seq[:-1]:
            prob = np.random.random()
            if prob < self.mask_prob:
                labels.append(item_idx)  # 학습에 사용
                masked_seq.append(self.n_items+1)
            else:
                labels.append(0)  # 학습에 사용 X
                masked_seq.append(item_idx)
        labels.append(seq[-1])
        masked_seq.append(self.n_items+1)
                
        # zero padding
        if seq.size < self.max_len:
            pad_len = self.max_len - seq.size
            masked_seq = [0] * pad_len + masked_seq
            labels = [0] * pad_len + labels
        
        return torch.LongTensor(masked_seq), torch.LongTensor(labels)