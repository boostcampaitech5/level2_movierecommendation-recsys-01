import os
import yaml
import wandb
import pickle
import pandas as pd
import numpy as np
from utils.train_valid import train_valid_split
from utils.preprocess import create_matrix_and_mappings
from utils.metric import recall_at_10
from utils.utils import korea_date_time
from model.ease import EASE


def main():
    # Path to the ease.yaml file
    config_path = "config/ease.yaml"

    # Load the contents of the YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    print("###############")
    print("Loading data.. \n")
    train_df = pd.read_csv(os.path.join(config["data_path"], "train_ratings.csv"))

    if config["cv"]:
        # WandB init
        wandb.login(key=config["wandb_key"])

        project_name = "MovieRec"
        user_name = config["user_name"]
        model_name = "EASE"
        date_time = korea_date_time()

        wandb_args = {
            "_lambda": config["_lambda"],
            "scale": config["scale"],
            "cv": config["cv"],
        }

        wandb.init(
            project=project_name,
            name=f"{user_name}-{model_name}-{date_time}",
            config=wandb_args,
        )

        print("###############")
        print("Splitting train / valid..\n")
        train_data, valid_data = train_valid_split(train_df, num_seq=4, num_ran=6)

        # Create a folder to store the datasets
        # folder_path =config['data_path']
        # os.makedirs(folder_path, exist_ok=True)

        # Save train data and valid data as .csv files
        # train_data.to_csv(os.path.join(folder_path, 'train_data.csv'), index=False)
        # valid_data.to_csv(os.path.join(folder_path, 'valid_data.csv'), index=False)

        print("###############")
        print("Evaluating with valid data..\n")
        X_cv, index_to_user, index_to_item = create_matrix_and_mappings(
            train_data, config["scale"]
        )
        model_cv = EASE(_lambda=config["_lambda"])
        model_cv.train(X_cv)
        result = model_cv.forward(X_cv[:, :])
        result[X_cv.nonzero()] = -np.inf

        # Extract top 10
        recommend_list = []
        for i in range(len(result)):
            sorted_indices = np.argsort(-result[i])
            for j in sorted_indices.tolist():
                for k in range(10):
                    recommend_list.append((index_to_user[i], index_to_item[j[k]]))

        pred_df = pd.DataFrame(recommend_list, columns=["user", "item"])

        # Evaluate Recall@10 performance
        recall_10 = recall_at_10(true_df=valid_data, pred_df=pred_df)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Recall@10:", recall_10)
        wandb.log({"Recall@10": recall_10})
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    # train with total data
    print("###############")
    print("Preprocessing..\n")
    X, _, _ = create_matrix_and_mappings(train_df, config["scale"])

    print("###############")
    print("Model: EASE\n")
    model = EASE(_lambda=config["_lambda"])

    print("###############")
    print("Training..\n")
    model.train(X)

    print("###############")
    print("Saving model..\n")
    save_folder = config["save_model_path"]
    os.makedirs(save_folder, exist_ok=True)

    with open(save_folder + "/model_params.pkl", "wb") as f:
        pickle.dump(model, f)

    print("###############")
    print("Saved successfully")


if __name__ == "__main__":
    main()
