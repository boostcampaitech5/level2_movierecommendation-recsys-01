import pandas as pd 
import os

def make_inter(path='../../data/train/'):
    
    df = pd.read_csv(path + 'train_ratings.csv')
    
    df.columns=['user_id:token','item_id:token','timestamp:float']
    outpath = f"dataset/recbole_train"
    
    os.makedirs(outpath, exist_ok=True)
    df.to_csv(os.path.join(outpath,"recbole_train.inter"),sep='\t',index=False)
    
def make_yaml(model_name):
    
    yaml_data = """
    USER_ID_FIELD: user_id
    ITEM_ID_FIELD: item_id
    TIME_FIELD: timestamp

    load_col:
        inter: [user_id, item_id, timestamp]
    """
    outpath = f"yaml"
    
    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath,f"{model_name}.yaml"),"w") as f:
        f.write(yaml_data)