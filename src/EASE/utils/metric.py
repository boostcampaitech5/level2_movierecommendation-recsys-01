def recall_at_10(true_df, pred_df):
    """
    Calculate Recall@10 metric for evaluating recommendation performance.

    Args:
        true_df (pandas.DataFrame): DataFrame containing true user-item interactions.
        pred_df (pandas.DataFrame): DataFrame containing predicted top-k items for each user.

    Returns:
        float: Mean Recall@10 score across all users.

    """
    # Create DataFrame of true interacted items for each user
    true_items = true_df.groupby('user')['item'].apply(set).reset_index(name='true_items')

    # Create DataFrame of predicted top-k items for each user
    pred_items = pred_df.groupby('user')['item'].apply(set).reset_index(name='pred_items')

    # Calculate recall@10 scores for each user
    recall_scores = []
    for _, row in true_items.iterrows():
        user = row['user']
        true_set = row['true_items']

        # Check if there are predicted items for the user
        pred_set = set(pred_items[pred_items['user'] == user]['pred_items'].values[0])
        intersection = true_set.intersection(pred_set)
        recall = len(intersection) / 10
        recall_scores.append(recall)

    # Calculate mean recall@10 across all users
    mean_recall = sum(recall_scores) / len(recall_scores)

    return mean_recall
