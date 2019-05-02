# https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
# https://www.youtube.com/watch?v=rGWBo0JGf50&t=7s
import random
import numpy as np
from Population import Population
from Agent import Agent
from os import path


class LudoPlayerGenetic:
    script_dir = path.dirname(__file__)
    file = '/pop_pool/simple_pop1.pool'
    Population population

    def __init__(self):
        None

    def load_pop(self):
        pop = np.load(self.script_dir + self.file + '.npz')
        self.population.pop = pop['ch']
        self.population.generation =

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
        np.random.choice(self.pop)

        return np.random.choice(np.argwhere(next_states != False))[0]


pop = Population()
pop.init_pop()
pop.save_pop()

player = LudoPlayerGenetic()
player.load_pop()
print(random.choice(pop.pop))
