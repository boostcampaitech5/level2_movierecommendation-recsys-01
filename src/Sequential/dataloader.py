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
    train_seq = list()
    valid_seq = list()
    valid_target = list()
    infer_seq = list()
    valid_cand = list()
    infer_cand = list()
    for user_idx, user_total in enumerate(tqdm(total)):
        # user_valid_target: 맨 뒤에서 5개, 중간에서 5개 추출
        user_valid_target = np.random.choice(user_total[:-(k//2)], (k//2), replace=False)
        user_valid_target = np.append(user_valid_target, user_total[-(k-k//2):])
        valid_target.append(user_valid_target)
        
        # user_total_train: user_valid_target 제외
        user_total_train = user_total[~np.isin(user_total, user_valid_target)]
        # max_len만큼 여러번 샘플링해서 train_seq에 추가
        for _ in range(1):
            user_train_seq_idx = np.random.choice(
                np.arange(0, user_total_train.size), min(user_total_train.size, max_len), replace=False)
            user_train_seq_idx = np.sort(user_train_seq_idx)
            train_sample = user_total_train[user_train_seq_idx]
            train_seq.append(train_sample)
        
        temp_valid_seq_idx = np.sort(np.random.choice(
            np.arange(0, user_total_train.size), min(user_total_train.size, max_len-k), replace=False))
        temp_valid_seq = user_total_train[temp_valid_seq_idx]
        user_valid_seq = np.zeros(temp_valid_seq.size+k, dtype=int)
        user_valid_seq[-k//2:] = n_items+1
        idx = np.sort(
            np.random.choice(np.arange(0, temp_valid_seq.size+(k//2)), k//2, replace=False))
        user_valid_seq[idx] = n_items+1
        user_valid_seq[user_valid_seq == 0] = temp_valid_seq
        if user_valid_seq.size < max_len :
            pad_len = max_len - user_valid_seq.size
            user_valid_seq = np.append([0]*pad_len, user_valid_seq)
        valid_seq.append(torch.tensor(user_valid_seq).unsqueeze(0))
        
        temp_infer_seq_idx = np.sort(np.random.choice(
            np.arange(0, user_total.size), min(user_total.size, max_len-k), replace=False))
        temp_infer_seq = user_total[temp_infer_seq_idx]
        user_infer_seq = np.zeros(temp_infer_seq.size+k, dtype=int)
        user_infer_seq[-k//2:] = n_items+1
        idx = np.sort(
            np.random.choice(np.arange(0, temp_infer_seq.size+(k//2)), k//2, replace=False))
        user_infer_seq[idx] = n_items+1
        user_infer_seq[user_infer_seq == 0] = temp_infer_seq
        if user_infer_seq.size < max_len :
            pad_len = max_len - user_infer_seq.size
            user_infer_seq = np.append([0]*pad_len, user_infer_seq)
        infer_seq.append(torch.tensor(user_infer_seq).unsqueeze(0))
        
        user_infer_cand = np.setdiff1d(np.arange(1, n_items+1), user_total)
        user_valid_cand = np.append(user_valid_target, user_infer_cand)
        valid_cand.append(user_valid_cand)
        infer_cand.append(user_infer_cand)
        
    data = {'train': train_seq,
            'valid': valid_seq,
            'valid_target': valid_target,
            'infer': infer_seq,
            'valid_cand': valid_cand,
            'infer_cand': infer_cand}
    
    return data, n_items, n_users, idx2item


class BERT4RecDataset(Dataset):
    def __init__(self,
                 train_data: pd.Series,
                 n_users: int,
                 n_items: int,
                 max_len: int,
                 k:int,
                 mask_prob: float):
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.max_len = max_len
        self.k = k
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, user_idx: int) -> Tuple[torch.tensor, torch.tensor]: 
        seq = self.train_data[user_idx]
        masked_seq = seq.copy()
        labels = np.zeros_like(seq)
        # for item_idx in seq[:-(self.k//2)]:
        #     prob = np.random.random()
        #     if prob < self.mask_prob:
        #         labels.append(item_idx)  # 학습에 사용
        #         masked_seq.append(self.n_items+1)
        #     else:
        #         labels.append(0)  # 학습에 사용 X
        #         masked_seq.append(item_idx)
        # labels.extend(seq[-(self.k//2):])
        # masked_seq.extend([self.n_items+1]*(self.k//2))
        
        mask_idx = np.random.choice(np.arange(0, seq.size-(self.k//2)), int(seq.size*self.mask_prob))
        # 중간 랜덤 5개
        masked_seq[mask_idx] = self.n_items+1
        # 마지막 5개
        masked_seq[-(self.k//2):] = self.n_items+1
        labels[mask_idx] = seq[mask_idx]
        labels[-(self.k//2):] = seq[-(self.k//2):]
                
        # zero padding
        if seq.size < self.max_len:
            pad_len = self.max_len - seq.size
            masked_seq = np.append([0] * pad_len, masked_seq)
            labels = np.append([0] * pad_len, labels)
        
        masked_seq = torch.LongTensor(masked_seq)
        labels = torch.LongTensor(labels)
        
        return masked_seq, labels