import implicit
import pandas as pd
from load_config import load_config
from load_data import load_data, create_user_item_map
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = load_config()
    ratings_data, sparse_matrix = load_data()

    model = implicit.als.AlternatingLeastSquares(
        factors=args.factors,  # The number of latent factors to compute
        regularization=args.regularization,
        iterations=args.iterations,
        calculate_training_loss=False,
        use_gpu=False,
    )

    print("model fit start")
    # https://benfred.github.io/implicit/api/models/cpu/als.html#implicit.cpu.als.AlternatingLeastSquares.fit
    model.fit(sparse_matrix)

    l = []
    user_id_to_user_map, item_id_to_item_map, _, _ = create_user_item_map(ratings_data)

    # submission
    for user_id in ratings_data["user_id"].unique():
        item, score = model.recommend(user_id, sparse_matrix[user_id], 10)
        original_user = user_id_to_user_map[user_id]

        for rec_item in item:
            original_item = item_id_to_item_map[rec_item]

            d = dict()
            d["user"] = original_user
            d["item"] = original_item
            l.append(d)

    pd.DataFrame(l).to_csv(Path.cwd() / "output.csv", index=False)
