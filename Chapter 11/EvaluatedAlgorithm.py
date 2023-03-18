from RecommenderMetrics import RecommenderMetrics
from EvaluationData import EvaluationData


class EvaluatedAlgorithm:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
        metrics = {}
        # compute accuracy
        if (verbose):
            print('Evaluating accuracy...')
        self.algorithm.fit(evaluationData.GetTrainSet())
        predictions = self.algorithm.test(evaluationData.GetTestSet())
        metrics['RMSE'] = RecommenderMetrics.RMSE(predictions)
        metrics['MAE'] = RecommenderMetrics.MAE(predictions)

        if (doTopN):
            # evaluate top-10 with Leave One Out testing
            if (verbose):
                print('Evaluating top-N with leave-one-out...')
            self.algorithm.fit(evaluationData.GetLOOCVTrainSet())
            leftOutPredictions = self.algorithm.test(
                evaluationData.GetLOOCVTestSet())
            # build predictions for all ratings not in the training set
            allPredictions = self.algorithm.test(
                evaluationData.GetLOOCVAntiTestSet())
            # compute top 10 recs for each user
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print('Computing hit-rate and rank metrics...')
            # see how often we recommended a movie the user actually rated
            metrics['HR'] = RecommenderMetrics.HitRate(
                topNPredicted, leftOutPredictions)
            # see how often we recommended a movie the user actually liked
            metrics['cHR'] = RecommenderMetrics.CumulativeHitRate(
                topNPredicted, leftOutPredictions)
            # compute ARHR
            metrics['ARHR'] = RecommenderMetrics.AverageReciprocalHitRank(
                topNPredicted, leftOutPredictions)

            # evaluate properties of recommendations on full training set
            if (verbose):
                print('Computing recommendations with full data set...')
            self.algorithm.fit(evaluationData.GetFullTrainSet())
            allPredictions = self.algorithm.test(
                evaluationData.GetFullAntiTestSet())
            topNPredicted = RecommenderMetrics.GetTopN(allPredictions, n)
            if (verbose):
                print('Analyzing coverage, diversity, and novelty...')
            # print user coverage with a minimum predicted rating of 4.0:
            metrics['Coverage'] = RecommenderMetrics.UserCoverage(
                topNPredicted, evaluationData.GetFullTrainSet().n_users, ratingThreshold=4.0)
            # measure diversity of recommendations:
            metrics['Diversity'] = RecommenderMetrics.Diversity(
                topNPredicted, evaluationData.GetSimilarities())

            # measure novelty (average popularity rank of recommendations):
            metrics['Novelty'] = RecommenderMetrics.Novelty(
                topNPredicted, evaluationData.GetPopularityRankings())

        if (verbose):
            print('Analysis complete.')

        return metrics

    def GetName(self):
        return self.name

    def GetAlgorithm(self):
        return self.algorithm
