from pyludo import LudoGame, StandardLudoPlayers
import matplotlib.pyplot as plt
import numpy as np
from os import path
from LudoPlayerGenetic import LudoPlayerGenetic
from Population import Population


def load_pop_size(gen=1, file='', script_dir=''):
    return len(np.load(script_dir + file + str(gen) + '.npy'))


def load_pop(population, gen=1, file='', script_dir=''):
    population.file = file
    population.script_dir = script_dir
    population.load_pop(gen)


def load_only_fitness(gen=1, file='fitness_', folder=''):
    script_dir = path.dirname(__file__) + '/' + folder + '/'
    return np.load(script_dir + file + str(gen) + '.npy')


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


def plot_pop200(to_gen=100, file='pop200_', folder='pop_pool', title='Random, sigma/4 '):

    script_dir = path.dirname(__file__) + '/' + folder + '/'
    fitness = []
    fitaxis = []
    fitmax = []
    pop_size = load_pop_size(gen=1, file=file, script_dir=script_dir)
    population = Population(pop_size)
    gens = range(1, to_gen + 1)
    for i in gens:
        load_pop(population, gen=i, file=file, script_dir=script_dir)
        assign_fitness(100, population=population)
        fitness = np.append(fitness, population.fitness)
        fitaxis = np.append(fitaxis, (np.zeros(pop_size, dtype=np.int) + i))
        fitmax = np.append(fitmax, population.fitness[np.argmax(population.fitness)])
    plt.figure()
    plt.plot(fitaxis, fitness, 'o', gens, np.sum(np.split(fitness / pop_size, len(gens)), axis=1))
    plt.ylabel('win rate')
    plt.xlabel('Generation')
    plt.legend(['Win-rate', 'Mean win-rate'], loc='center right')
    plt.suptitle(title)


plot_gen = 500

#plot_pop200(to_gen=plot_gen, file='pop200_', folder='pop_poolt', title='t ')

#plot_pop200(to_gen=plot_gen, file='pop_pool_self_', folder='pop_pool_self', title='Self ')
#plot_pop200(to_gen=plot_gen, file='pop200_more_lossy_muta_sigma_', folder='pop_pool_more_lossy_10_muta_40_sigma_4', title='Random, pop_pool_more_lossy_10_muta_40_sigma_4 ')
#plot_pop200(to_gen=plot_gen, file='pop200_lossy_muta_sigma_', folder='pop_pool_lossy_7_muta_15_sigma_2', title='Random, pop_pool_lossy_7_muta_15_sigma_2 ')
#plot_pop200(to_gen=plot_gen, file='pop200_sigma2_', folder='pop_pool_sigma_2', title='Random, half sigma ')
#plot_pop200(to_gen=plot_gen, file='pop200_lossy_', folder='pop_pool_more_loss', title='Random, High selection ')

plt.figure()
plt.plot(load_only_fitness(gen=500, file='fitness_', folder='pop_pool'), 'o', load_only_fitness(gen=500, file='fitness_t_', folder='pop_pool_self'), 'ro')
plt.ylabel('win rate')
plt.xlabel('Population')
plt.legend(['random', 'self'], loc='lower right')
plt.suptitle('Random vs. self')
plt.xlim([0, 100])

#plot_pop200(to_gen=plot_gen, file='pop200_', folder='pop_pool', title='Random ')

#plot_pop200(to_gen=plot_gen, file='pop40_', folder='pop_pool_small', title='small')
#plot_pop200(to_gen=plot_gen, file='pop40_all_', folder='pop_pool_small_all', title='small')

plt.show()
'''
f = [[1, 2], [3, 4]]
f = np.append(f, [5, 6])

print(f)
print(np.sum(np.split(f, 3), axis=1))
'''
