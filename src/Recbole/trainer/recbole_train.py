import os 
import pandas as pd 
import numpy as np 
import torch 

from recbole.model.general_recommender.bpr import BPR
from recbole.model.general_recommender.ease import EASE
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.model.general_recommender.multivae import MultiVAE
from recbole.model.general_recommender.neumf import NeuMF

from recbole.model.context_aware_recommender.ffm import FFM
from recbole.model.context_aware_recommender.fm import FM

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.quick_start import load_data_and_model, run_recbole
from recbole.utils.case_study import full_sort_topk

from logging import getLogger

def general_train(model_name='ease'):
    logger = getLogger()
    
    if model_name=='lightgcn':
        config = Config(model='LightGCN',dataset='general_train',config_file_list=[f"general_yaml/{model_name}.yaml"])
    
    elif model_name == 'multivae':
        config = Config(model='MultiVAE',dataset='general_train',config_file_list=[f"general_yaml/{model_name}.yaml"])
        
    elif model_name == 'neumf':
        config = Config(model='NeuMF',dataset='general_train',config_file_list=[f"general_yaml/{model_name}.yaml"])
    else:
        config = Config(model=model_name.upper(),dataset='general_train',config_file_list=[f"general_yaml/{model_name}.yaml"])
    
    config['show_progress'] = False 
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    logger.info(config)
    
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    init_seed(config['seed'], config['reproducibility'])
    if model_name =='ease':
        model = EASE(config, train_data.dataset).to(config['device'])
    elif model_name == 'bpr':
        model = BPR(config, train_data.dataset).to(config['device'])
    elif model_name == 'lightgcn':
        model = LightGCN(config, train_data.dataset).to(config['device'])
    elif model_name == 'multivae':
        model = MultiVAE(config, train_data.dataset).to(config['device'])
    elif model_name == 'neumf':
        model = NeuMF(config, train_data.dataset).to(config['device'])
    else:
        raise NotImplementedError(f"model {model_name} not implemented")

    logger.info(model)
    
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    
    test_result = trainer.evaluate(test_data,load_best_model=True,show_progress=config['show_progress'])
    
    print(test_result)

def context_train(model_name='ffm'):
    logger = getLogger()

    if model_name=='ffm':
        config = Config(model='FFM', dataset="context_train", config_file_list=[f'context_yaml/ffm.yaml'])
    elif model_name=='fm':
        config = Config(model='FM', dataset="context_train", config_file_list=[f'context_yaml/fm.yaml'])
    else:
        raise NotImplementedError(f"model {model_name} not implemented")
    
    config['epochs'] = 100
    config['show_progress'] = False
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)

    logger.info(config)
    
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    if model_name=='ffm':
        model = FFM(config, train_data.dataset).to(config['device'])
    elif model_name=='fm':
        model = FM(config, train_data.dataset).to(config['device'])
        
    logger.info(model)
    
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )