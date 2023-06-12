import os
import csv
import yaml
import pickle
import numpy as np
import pandas as pd
from model.ease import EASE
from utils.utils import korea_date_time
from utils.preprocess import create_matrix_and_mappings

def main():
    # Path to the ease.yaml file
    config_path = 'config/ease.yaml'
    
    # Load the contents of the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    train_df = pd.read_csv(os.path.join(config['data_path'], 'train_ratings.csv'))
    
    X, index_to_user, index_to_item = create_matrix_and_mappings(train_df)
    
    print('###############')
    print('Load saved model.. \n')
    with open(config['save_model_path'] + "/model_params.pkl", "rb") as f:
        model = pickle.load(f)
    
    print('###############')
    print('Predicting.. \n')
    result = model.forward(X[:, :])
    #To remove information that has already been evaluated
    result[X.nonzero()] = -np.inf

    #Extract and save the top 10
    recommend_list=[]
    for i in range(len(result)):
        sorted_indices = np.argsort(-result[i])
        for j in sorted_indices.tolist():
            for k in range(10):
                recommend_list.append((index_to_user[i], index_to_item[j[k]]))

    # Save inference
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    date_time = korea_date_time()
    filename = filename = os.path.join(output_folder, f'output_{date_time}.csv')
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user', 'item'])
        writer.writerows(recommend_list)

    print(f"Recommendations saved to '{filename}'")

if __name__ == "__main__":
    main()