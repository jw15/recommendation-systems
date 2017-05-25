from item_item_prototype import make_cos_sim_and_neighborhoods, get_ratings_data
from ItemItemRecommender import ItemItemRecommender
# from ItemItemRecommender import pred_one_user
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    ratings_data_contents, ratings_mat = get_ratings_data()
    user = ratings_data_contents['user']
    recommender = ItemItemRecommender(ratings_mat)
    recommender._set_neighborhoods()
    user_1_preds = recommender._pred_one_user(user_id=1)
    # cos_sim, nbrhoods = make_cos_sim_and_neighborhoods(ratings_mat, neighborhood_size=75)
    # user_1_preds = pred_one_user(cos_sim, nbrhoods, ratings_mat, user_id=1)
    # Show predicted ratings for user #1
    # print(user_1_preds)
