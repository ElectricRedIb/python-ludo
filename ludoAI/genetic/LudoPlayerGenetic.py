# https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
# https://www.youtube.com/watch?v=rGWBo0JGf50&t=7s
import random
import numpy as np
from os import path


class LudoPlayerGenetic:
    name = 'genetic'

    def __init__(self, loading_agent):
        self.agent = loading_agent

    def load_agent(self, loading_agent):
        self.agent = loading_agent  # self.population.pop[np.random.choice(np.arange(len(self.population.pop)), size=1, replace=False, p=None)[0]]
        # print(self.agent)

    def play(self, state, dice_roll, next_states):
        """
        :param state:
            current state relative to this player
        :param dice_roll:
            [1, 6]
        :param next_states:
            np array of length 4 with each entry being the next state moving the corresponding token.
            False indicates an invalid move. 'play' won't be called, if there are no valid moves.
        :return:
            index of the token that is wished to be moved. If it is invalid, the first valid token will be chosen.
            np.random.choice(np.argwhere(next_states != False))[0]
        """
        valid_moves = np.argwhere(next_states != False)
        go_on_board = np.zeros(4, dtype=int)
        go_in_goal = np.zeros(4, dtype=int)
        move = np.zeros(4, dtype=float)
        hit_home = np.zeros(4, dtype=int)

        for i in valid_moves:
            go_on_board[i[0]] = next_states[i[0]][0][i[0]] == 1
            go_in_goal[i[0]] = next_states[i[0]][0][i[0]] == 99
            move[i[0]] = next_states[i[0]][0][i[0]] / 99
            hit_home[i[0]] = np.sum(next_states[1:] == -1) > np.sum(state[1:] == -1)  # sum(state.state)  # - sum(state[0])  # - sum(next_states[i[0]].state)  # - sum(next_states[i[0]][0])

        decision_matrix = np.matrix([go_on_board * self.agent[0]] + [go_in_goal * self.agent[1]] + [move * self.agent[2]] + [hit_home * self.agent[3]])

        decision = decision_matrix.sum(axis=0)   # [decision_matrix[i[0]] for i in valid_moves]

        choice = np.argmax(decision.flat[valid_moves])  # random.choice(np.argwhere(next_states != False))[0]

        return valid_moves[choice][0]
