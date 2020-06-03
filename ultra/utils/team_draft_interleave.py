import numpy as np


class TeamDraftInterleaving():
    def __init__(self, ):
        pass

    def next_index_to_add(self, inter_result, inter_n, ranking, index):
        while index < ranking.shape[0] and np.any(
                ranking[index] == inter_result[:inter_n]):
            index += 1
        return index

    def interleave(self, rankings):
        self.n_rankers = rankings.shape[0]
        k = rankings.shape[1]
        teams = np.zeros(k, dtype=np.int32)
        multileaved = np.zeros(k, dtype=np.int32)

        multi_i = 0
        while multi_i < k and np.all(
                rankings[1:, multi_i] == rankings[0, multi_i]):
            multileaved[multi_i] = rankings[0][multi_i]
            teams[multi_i] = -1
            multi_i += 1

        indices = np.zeros(self.n_rankers, dtype=np.int32) + multi_i
        assignment = np.arange(self.n_rankers)
        assign_i = self.n_rankers
        while multi_i < k:
            if assign_i == self.n_rankers:
                np.random.shuffle(assignment)
                assign_i = 0

            rank_i = assignment[assign_i]
            indices[rank_i] = self.next_index_to_add(multileaved, multi_i,
                                                     rankings[rank_i, :],
                                                     indices[rank_i])
            multileaved[multi_i] = rankings[rank_i, indices[rank_i]]
            teams[multi_i] = rank_i
            indices[rank_i] += 1
            multi_i += 1
            assign_i += 1

        self.teams = teams
        return multileaved

    def infer_winner(self, clicks):
        click_matrix = np.array(
            [(self.teams[:len(clicks)] == i) * clicks for i in range(self.n_rankers)])
        # print (click_matrix)
        ranker_clicks = np.sum(click_matrix, axis=1)
        return ranker_clicks / (np.sum(ranker_clicks) + 0.0000001)
