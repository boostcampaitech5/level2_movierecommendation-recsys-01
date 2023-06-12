import os
import yaml
import pickle
import pandas as pd
from utils.preprocess import create_matrix_and_mappings
from model.ease import EASE

def main():
    # Path to the ease.yaml file
    config_path = 'config/ease.yaml'

    # Load the contents of the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print('###############')
    print('Loading data.. \n')
    train_df = pd.read_csv(os.path.join(config['data_path'], 'train_ratings.csv'))

    print('###############')
    print('Preprocessing..\n')
    X, _, _ = create_matrix_and_mappings(train_df)

    print('###############')
    print('Model: EASE\n')
    model = EASE(_lambda=config['_lambda'])
    
    print('###############')
    print('Training..\n')
    model.train(X)

    print('###############')
    print('Saving model..\n')
    save_folder = config['save_model_path']
    os.makedirs(save_folder, exist_ok=True)

    with open(save_folder + "/model_params.pkl", "wb") as f:
        pickle.dump(model, f)

    print('###############')
    print('Saved successfully')

if __name__ == '__main__':
    main()
