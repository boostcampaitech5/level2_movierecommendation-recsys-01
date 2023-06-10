import numpy as np
import torch
from tqdm import tqdm
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import pandas as pd
import os

def inference(model_path='EASE-Jun-10-2023_14-11-23.pth'):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=os.path.join('saved',model_path))
    
    del train_data, valid_data
    
    submission = pd.read_csv('../../data/eval/sample_submission.csv')
    sub_user_idx = submission['user'].unique()
    sub_user_idx = np.array(sub_user_idx,dtype=str)
    uid_series = dataset.token2id(dataset.uid_field, sub_user_idx)
    total_topk_score, total_topk_iid_list = torch.zeros_like(torch.Tensor(31360, 10)), torch.zeros_like(torch.Tensor(31360, 10))
    
    for idx in tqdm(range(0,len(uid_series))):
        topk_score, topk_iid_list = full_sort_topk(np.array([uid_series[idx]]),model,test_data,10,config['device'])
        total_topk_score[idx] = topk_score
        total_topk_iid_list[idx] = topk_iid_list
        
    int_iid = total_topk_iid_list.to(torch.int64)
    external_item_list = dataset.id2token(dataset.iid_field, int_iid.cpu())
    external_item_list = external_item_list.flatten()
    df = pd.DataFrame({'user': np.repeat(sub_user_idx, 10), 'item': external_item_list})
    df.to_csv("submission.csv",index=False)