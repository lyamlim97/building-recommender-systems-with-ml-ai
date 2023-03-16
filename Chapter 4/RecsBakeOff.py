from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np


def LoadMovieLensData():
    ml = MovieLens()
    print('Loading movie ratings...')
    data = ml.loadMovieLensLatestSmall()
    print('\nComputing movie popularity ranks so we can measure novelty later...')
    rankings = ml.getPopularityRanks()
    return (data, rankings)


np.random.seed(0)
random.seed(0)

# load up common data set for the recommender algorithms
(evaluationData, rankings) = LoadMovieLensData()

# construct an Evaluator object
evaluator = Evaluator(evaluationData, rankings)

# add in an SVD recommender
SVDAlgorithm = SVD(random_state=10)
evaluator.AddAlgorithm(SVDAlgorithm, 'SVD')

# add in a random recommender
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, 'Random')

# compare algorithms
evaluator.Evaluate(True)
