import os 
import pandas as pd 
import numpy as np 
import torch 

from recbole.model.general_recommender.bpr import BPR
from recbole.model.general_recommender.ease import EASE

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, Interaction
from recbole.utils import init_logger, get_trainer, get_model, init_seed, set_color
from recbole.quick_start import load_data_and_model, run_recbole
from recbole.utils.case_study import full_sort_topk

from logging import getLogger

def train(model_name='ease'):
    logger = getLogger()
    
    config = Config(model=model_name.upper(),dataset='recbole_train',config_file_list=[f"yaml/{model_name}.yaml"])
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
