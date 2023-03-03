from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData


def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)


ml, data, rankings = LoadMovieLensData()

evalData = EvaluationData(data, rankings)

# pick user based or item based cf (user/item)
mode = 'item'

# train on leave-One-Out train set
trainSet = evalData.GetLOOCVTrainSet()
if mode == 'user':
    sim_options = {'name': 'cosine',
                   'user_based': True
                   }
elif mode == 'item':
    sim_options = {'name': 'cosine',
                   'user_based': False
                   }
else:
    print('Please select correct CF type')

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

leftOutTestSet = evalData.GetLOOCVTestSet()

# build up dict to lists of (int(movieID), predictedrating) pairs
topN = defaultdict(list)
k = 10

for uiid in range(trainSet.n_users):
    if mode == 'user':
        # get top N similar users to this one
        similarityRow = simsMatrix[uiid]

        similarUsers = []
        for innerID, score in enumerate(similarityRow):
            if (innerID != uiid):
                similarUsers.append((innerID, score))

        kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])

        # get the stuff they rated, and add up ratings for each item, weighted by user similarity
        candidates = defaultdict(float)
        for similarUser in kNeighbors:
            innerID = similarUser[0]
            userSimilarityScore = similarUser[1]
            theirRatings = trainSet.ur[innerID]
            for rating in theirRatings:
                candidates[rating[0]] += (rating[1] / 5.0) * \
                    userSimilarityScore

    elif mode == 'item':
        userRatings = trainSet.ur[uiid]

        kNeighbors = heapq.nlargest(k, userRatings, key=lambda t: t[1])

        candidates = defaultdict(float)
        for itemID, rating in kNeighbors:
            similarityRow = simsMatrix[itemID]
            for innerID, score in enumerate(similarityRow):
                candidates[innerID] += score * (rating / 5.0)

    else:
        print('Please select correct CF type')

    # build a dictionary of stuff the user has already seen
    watched = {}
    for itemID, rating in trainSet.ur[uiid]:
        watched[itemID] = 1

    # get top-rated items from similar users:
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            movieID = trainSet.to_raw_iid(itemID)
            topN[int(trainSet.to_raw_uid(uiid))].append((int(movieID), 0.0))
            pos += 1
            if (pos > 40):
                break

# measure
print("HR", RecommenderMetrics.HitRate(topN, leftOutTestSet))
