import os
import json
import torch
from torch.utils.data import DataLoader
import wandb

from utils import get_timestamp, set_seeds
from dataloader import load_data, process_data, BERT4RecDataset
from model import BERT4Rec
from trainer import run, inference
    
    
def main():
    print("Load Configuration File.")
    with open(os.path.join(os.curdir, 'config.json'), 'r') as f:
        CONFIG = json.load(f)
    
    data_dir = CONFIG['data_dir']
    model_dir = CONFIG['model_dir']
    output_dir = CONFIG['output_dir']
    os.makedirs(name=model_dir, exist_ok=True)
    os.makedirs(name=output_dir, exist_ok=True)
    set_seeds(CONFIG['seed'])
    
    if CONFIG['device'] == 'cuda' and torch.cuda.is_available() == True :
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        CONFIG['device'] = 'cpu'
    timestamp = get_timestamp()
    
    if CONFIG['logging'] == True:
        print("Start Logging with WandB")
        with open(os.path.join(os.curdir, 'key.txt'), 'r') as f:
            key = f.readline()
        wandb.login(key=key)
        wandb.init(
            project="movie_rec",
            name=f"{'BERT4Rec'}-{timestamp}",
            config=CONFIG,
        )
        
    for config, value in CONFIG.items():
        print(f"{config}: {value}")
        
    print("Load and Process Data.")
    train_df, sub_df = load_data(data_dir)
    data, n_items, n_users, idx2item = process_data(train_df,
                                                    CONFIG['max_len'],
                                                    CONFIG['k'],
                                                    CONFIG['n_samples'],
                                                    CONFIG['tail_ratio'])
    
    print("Create Dataset and Dataloader.")
    dataset = BERT4RecDataset(data['train'],
                              n_users,
                              n_items,
                              CONFIG['max_len'],
                              CONFIG['k'],
                              CONFIG['mask_prob'])
    data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    print("Create BERT4Rec Model.")
    model = BERT4Rec(n_items,
                     CONFIG['embed_dim'],
                     CONFIG['max_len'],
                     CONFIG['n_layers'],
                     CONFIG['n_heads'],
                     CONFIG['pffn_hidden_dim'],
                     CONFIG['unidirection'],
                     CONFIG['dropout_rate'],
                     device=device).to(device)

    print(f"Start Training for {CONFIG['n_epochs']} Epochs.")
    best_recall, best_epoch = run(model,
                                  data_loader,
                                  data['valid'],
                                  data['valid_cand'],
                                  CONFIG['k'],
                                  CONFIG['n_epochs'],
                                  CONFIG['lr'],
                                  CONFIG['max_patience'],
                                  CONFIG['logging'],
                                  model_dir,
                                  timestamp)
    
    for config, value in CONFIG.items():
        print(f"{config}: {value}")
    result = {'best_recall': best_recall, 'best_epoch': best_epoch}
    result_config = {'result': result, 'config': CONFIG}
    with open(f"{model_dir}/bert4rec_{timestamp}.json", "w") as f:
        json.dump(result_config, f)
        
    print("Make Inference.")
    inference(model,
              data['infer'],
              data['infer_cand'],
              CONFIG['k'],
              sub_df,
              idx2item,
              model_dir,
              output_dir,
              timestamp)
    # CONFIG['tail_ratio'] += 0.25
    # with open(os.path.join(os.curdir, 'config.json'), 'w') as f:
    #     json.dump(CONFIG, f)
    wandb.finish()


if __name__ == "__main__" :
    # for _ in range(5):
    #     main()
    main()
