from pyludo import LudoGame, StandardLudoPlayers
import numpy as np
import time
from genetic.LudoPlayerGenetic import LudoPlayerGenetic
from genetic.Population import Population


# ----------------------------------------------------------
population = Population(200)
# ----------------------------------------------------------


def tournament(generations=100, fitness_games=200):

    def assign_fitness(amount_of_games=100):
        # print('Assigning fitness to the population')
        agent_idx = population.get_fitless_agent_idx()
        while agent_idx != -1:
            GA_agent = population.pop[agent_idx]

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
            # print('win distribution:', fitness / amount_of_games)
            population.fitness_add(fitness_score=fitness[0] / amount_of_games, index_of_chromosomes=agent_idx)
            agent_idx = population.get_fitless_agent_idx()
    population.save_pop()
    assign_fitness(amount_of_games=fitness_games)
    for i in range(generations):
        print('Generation: ', i)
        population.breed(4)
        assign_fitness(amount_of_games=fitness_games)
        population.save_pop()
        population.save_fitness()
        population.reset_fitness()
        '''
        if population.generation % 10 == 0:
            population.save_fitness()
            normal_game(number_of_runs=100)
        '''


def normal_game(number_of_runs=100):

    GA_player = population.pop[population.get_best_agent()]  # population.pop[population.get_random_agent(1)[0]]
    players = [LudoPlayerGenetic(GA_player)] + [StandardLudoPlayers.LudoPlayerRandom() for _ in range(3)]
    for i, player in enumerate(players):
        player.id = i
        # print(player)

    score = np.zeros(4, dtype=np.int)

    n = number_of_runs

    start_time = time.time()
    for i in range(n):
        '''
        for idx in range(4):
            if players[idx].id == 0:
                players[idx].load_agent(population.pop[population.get_best_agent()])
        '''
        np.random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1
        # print('Game ', i, ' done')
    duration = time.time() - start_time
    print('win distribution:', score / number_of_runs)
    print('games per second:', n / duration)


# population.load_pop(500)
#tournament(generations=1, fitness_games=100)
# population.load_pop(15)
# print(population.fitness[population.get_best_agent()])
# normal_game(number_of_runs=100)

'''
population.load_pop(500)
normal_game(number_of_runs=200)

for i in range(100):
    tournament(total_fighters=4, amount_of_tournament=4, amount_of_games=10)
    print('Generation: ', population.generation)
    # print(population.pop)
    if i % 10 == 0:
        normal_game(number_of_runs=100)
        # print(population.pop)
normal_game(number_of_runs=200)
'''
'''
population.load_pop(3)
print(population.pop)
population.load_pop(4)
print(population.pop)
for _ in range(5):
    print(population.tournament(1))
'''
'''
for i in [2, 3]:
    population.load_pop(i)
    print(population.pop)
    # normal_game(number_of_runs=100)
'''
