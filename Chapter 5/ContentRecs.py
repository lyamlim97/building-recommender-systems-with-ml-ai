from MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np


def LoadMovieLensData():
    ml = MovieLens()
    print('Loading movie ratings...')
    data = ml.loadMovieLensLatestSmall()
    print('\nComputing movie popularity ranks so we can measure novelty later...')
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)


np.random.seed(0)
random.seed(0)

# load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

# construct an Evaluator object
evaluator = Evaluator(evaluationData, rankings)

# add in a content KNN recommender
contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, 'ContentKNN')

# add in a random recommender
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, 'Random')

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
