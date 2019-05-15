from pyludo import LudoGame, StandardLudoPlayers
import matplotlib.pyplot as plt
import numpy as np
from os import path
from LudoPlayerGenetic import LudoPlayerGenetic
from Population import Population


def load_pop_size(gen=1):
    return len(np.load(script_dir + file + str(gen) + '.npy'))


def load_pop(population, gen=1):
    population.file = file
    population.script_dir = script_dir
    population.load_pop(gen)


def assign_fitness(amount_of_games, population):
    # print('Assigning fitness to the population')
    agent_idx = population.get_fitless_agent_idx()
    while agent_idx != -1:
        GA_agent = population.pop[agent_idx]
        # print('Calculating fitness for agent: ', agent_idx)
        # print(GA_agent)
        players = [LudoPlayerGenetic(GA_agent)] + [StandardLudoPlayers.LudoPlayerRandom() for _ in range(3)]
        for id, player in enumerate(players):
            player.id = id

        n = amount_of_games
        fitness = np.zeros(4, dtype=np.int)
        # print(agent_idx)
        for i in range(n):

            np.random.shuffle(players)
            ludoGame = LudoGame(players)
            winner = ludoGame.play_full_game()
            fitness[players[winner].id] += 1
        population.fitness_add(fitness_score=fitness[0] / amount_of_games, index_of_chromosomes=agent_idx)
        agent_idx = population.get_fitless_agent_idx()


file = 'test_pop'
script_dir = path.dirname(__file__) + '/pop_pool_test/'

pop_size = load_pop_size(gen=1)
population = Population(pop_size)
load_pop(population, gen=1)
assign_fitness(5, population=population)
plt.plot(sum(population.fitness / pop_size))
plt.show()
