# https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3


import random
import numpy as np


class LudoPlayerGenetic:
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
        """

        return random.choice(np.argwhere(next_states != False))[0]

    def init_pop(self):
        None

    def fitness(self, pop):
        None

    def breed(self, pop):
        None

    def crossover(self, parent_a, parent_b):
        None

    def mutaion(self, child):
        None

    def execute(self, pop):
        None
