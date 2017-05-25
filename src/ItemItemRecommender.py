from item_item_prototype import make_cos_sim_and_neighborhoods, get_ratings_data
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

class ItemItemRecommender(object):
    def __init__(self, ratings_mat, user, neighborhood_size=75):
        '''
        Initialize the parameters of the model.
        '''
        self.neighborhood_size = neighborhood_size
        self.ratings_mat = ratings_mat
        # self.items_cos_sim, self.neighborhood = _set_neighborhoods(self)
        self.neighborhood = None
        self.items_cos_sim = None
        self.user = user

    def fit(self):
        '''
        Implement the model and fit it to the data passed as an argument.

        Store objects for describing model fit as class attributes.
        '''
        return self._fit(self.ratings_mat)

    def _set_neighborhoods(self):
        '''
        Get the items most similar to each other item.

        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.

        You will call this in your fit method.
        '''
        self.items_cos_sim, self.neighborhood = make_cos_sim_and_neighborhoods(self.ratings_mat, self.neighborhood_size)
        return self.items_cos_sim, self.neighborhood

    def _pred_one_user(self, user_id):
        '''
        Accept user id as arg. Return the predictions for a single user.

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        n_items = self.ratings_mat.shape[1]
        items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
        # Just initializing so we have somewhere to put rating preds
        output = np.zeros(n_items)
        for item_to_rate in range(n_items):
            relevant_items = np.intersect1d(self.neighborhood[item_to_rate], items_rated_by_this_user, assume_unique=True)
            # assume_unique speeds up intersection op
            output[item_to_rate] = ((self.ratings_mat[user_id, relevant_items] * \
                self.items_cos_sim[item_to_rate, relevant_items])) / \
                (self.items_cos_sim[item_to_rate, relevant_items].sum())
        return output

    def pred_all_users(self):
        '''
        Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).

        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        alluser_list = []
        for user_id in self.user:
            alluser_list.append(pred_one_user(user_id))
        alluser_arr = np.array(alluser_list)
        return alluser_arr.T

    def top_n_recs(self):
        '''
        Take user_id argument and number argument.

        Return that number of items with the highest predicted ratings, after
        removing items that user has already rated.
        '''
        pass
