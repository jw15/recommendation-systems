# Building an Item-Based collaborative filtering engine

You have received a small prototype for an item-item based recommender, and your job is to turn it into a python class that can be used as the basis for a recommendation engine on a small website.

You will be using the [MovieLens](http://grouplens.org/datasets/movielens/) dataset for this sprint. This data contains movie ratings on a scale from 1-5 in the [u.data](data/u.data) file.  There are other files we won't be using at this time, and they are described in the [readme](data/README).

You can also use this small toy dataset for verifying that your code is running properly.

```python
ratings_mat = sparse.lil_matrix(np.array([[4, 0, 0, 5, 1, 0, 0],
                                          [5, 5, 4, 0, 0, 0, 0],
                                          [0, 0, 0, 2, 4, 5, 0],
                                          [0, 3, 0, 0, 0, 0, 3]]))
```

1.  Review `item_item_prototype.py` and then run it. There are a few serious problems with the ratings shown.

	Come up with a hypothesis for why we are getting `nan`'s in our predictions. Write your explanation for why we are getting `nan`'s in a file called `explanations.txt`.

There are nan's in the predictions because there are zero values in the denominator for the final calculation (sum of similarities * ratings) / sum of similarities.

2.  Move all of our functionality from the prototype to a class called `ItemItemRecommender`. `src/ItemItemRecommender.py` has a basic skeleton for a class.  The skeleton follows the convention of sklearn, where the `__init__` function describes the parameters of the model (in this case, it takes the neighborhood size as an argument and sets `self.neighborhood_size`).  Then a `fit` method of the class takes the data (in this case `ratings_mat`) and sets everything else in the `make_cos_sim_and_neighborhoods` function.

	Do not call the `fit` method from the constructor.  To match the common convention, let the user of the class do that.

	Add code outside the class so you can run this file and get the same output you got in the prototype.

3.  Modify `pred_one_user` to replace the missing values with something numerical. `numpy.nan_to_num` is a good option for this.

4.  Add a `pred_all_users` method. The output should be a 2-dimensional array with the same shape as the matrix of actual ratings. Accomplish this by calling your `pred_one_user` method repeatedly.

5.  In a live setting, we'd have to be able to make recommendations for a single user quickly. Add an optional argument to pred_one_user that indicates whether to print the running time.  This argument should default to False.  Add `from time import time` to the imports part of this file. Add the functionality in `pred_one_user` to print running time when requested. Which configurations of your recommender result in shorter times? What things could you compute in advance to make your recommender more performant? What are the downsides of this pre-computing?

6.  Add a method to return the indices (column numbers) of the top n recommendations for a user.  Exclude items the user has already rated. This method should take the user_id and n as arguments.  It should return a list. Use `np.argsort`. Then be careful to preserve that ordering when you remove the already-rated items.

7.  What do you think are the major shortcomings of the current recommender?  Write your answer in `explanations.txt`. If you have any ideas, suggest possible remedies.

## Bonus
You've built something cool. Play with it. For example, get recommendations for a user and see the predicted ratings for those recommendations. Try recommenders with differing neighborhood sizes. What happens to the distribution of recommendations as neighborhood size increases? Don't put anything from this step in your pull request.
