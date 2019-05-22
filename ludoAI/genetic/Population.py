from pyludo import LudoGame, StandardLudoPlayers
import numpy as np
import random
from tempfile import TemporaryFile
from os import path


class Population:
    # pop_size = 200
    file = 'pop_pool_self_'  # '/pop_pool/4g_self_pop'
    script_dir = path.dirname(__file__) + '/pop_pool_self/'
    generation = 0
    amount_of_genes = 4
    mu = 0
    sigma = 1

    def __init__(self, pop_size=200):  # normal 200
        self.pop_size = pop_size
        self.pop = [np.random.normal(self.mu, self.sigma, self.amount_of_genes) for _ in range(self.pop_size)]
        self.fitness = np.zeros(self.pop_size, dtype=float) - 1.0

    def load_pop(self, load_generation):

        if path.exists(self.script_dir + self.file + str(load_generation) + '.npy'):
            self.pop = np.load(self.script_dir + self.file + str(load_generation) + '.npy')
            self.generation = load_generation
            if path.exists(self.script_dir + 'fitness_' + str(load_generation) + '.npy'):
                self.fitness = np.load(self.script_dir + 'fitness_' + str(load_generation) + '.npy')
            else:
                self.fitness = np.zeros(self.pop_size, dtype=float) - 1.0
            # print('loaded population gen', self.generation)

    def fitness_add(self, fitness_score, index_of_chromosomes):
        self.fitness[index_of_chromosomes] = fitness_score

    def get_fitless_agent_idx(self):
        enum_fitness = list(enumerate(self.fitness))
        sorted_fitness = sorted(enum_fitness, key=lambda x: x[1])
        if sorted_fitness[0][1] != -1:
            return -1
        return sorted_fitness[0][0]

    def breed(self, create_agents=10):
        if create_agents % 2:
            print('Create agents must be dividable by 2')
            return 0

        def get_fitness_p():
            return self.fitness / sum(self.fitness)

        def agents_to_breed(to_breed, fitness_p):
            return np.random.choice(np.arange(len(self.pop)), size=to_breed, replace=False, p=fitness_p)
        fitness_p = np.array(get_fitness_p())

        breeders = agents_to_breed(to_breed=create_agents, fitness_p=fitness_p)
        # parents_idx = breeders[np.argsort(fitness_p[breeders])]
        parents_idx = np.split(breeders, int(create_agents / 2))
        children = np.array([], dtype=int)
        for parent in parents_idx:
            children = np.append(self.crossover(parent_a=self.pop[parent[0]], parent_b=self.pop[parent[1]]), children)

        children = np.split(children, create_agents)
        self.execute(new_agents=children, fitness_p=fitness_p)

        self.generation = self.generation + 1

    def crossover(self, parent_a, parent_b):
        mask = np.random.rand(self.amount_of_genes) < 0.5
        child_a = parent_a * mask + parent_b * np.invert(mask)
        child_b = parent_b * mask + parent_a * np.invert(mask)
        child_a = self.mutation(child=child_a, mutation_rate=0.1)
        child_b = self.mutation(child=child_b, mutation_rate=0.1)
        return [child_a, child_b]

        '''
        The mutation rate is set to low in GA because high
        mutation rates convert GA to a primitive random search.
        --2019_Book_EvolutionaryAlgorithmsAndNeura

        page 47
        '''

    def mutation(self, child=[], mutation_rate=0.1):
        mask = np.random.rand(self.amount_of_genes) < mutation_rate
        mutation = [i for i, x in enumerate(mask) if x]
        for i in mutation:
            child[i] = np.random.normal(child[i], self.sigma / 4, self.amount_of_genes)[0]
        return child

    def execute(self, new_agents, fitness_p):
        # print('new agents', new_agents)
        replace_agent_idxs = np.arange(self.pop_size)[np.argsort(fitness_p)]
        for i in range(len(new_agents)):
            self.pop[replace_agent_idxs[i]] = new_agents[i]
            self.fitness[replace_agent_idxs[i]] = -1

    def save_pop(self):
        pop_to_save = self.pop
        np.save(self.script_dir + self.file + str(self.generation), pop_to_save)

    def save_fitness(self):
        fit_to_save = self.fitness
        np.save(self.script_dir + 'fitness_' + str(self.generation), fit_to_save)

    def get_random_agent(self, amount_of_agents=1):
        return np.random.choice(np.arange(len(self.pop)), size=amount_of_agents, replace=False, p=None)

    def get_best_agent(self):
        return np.argmax(self.fitness)

    def reset_fitness(self):
        self.fitness = np.zeros(self.pop_size, dtype=float) - 1.0


'''
list1 = list(enumerate(list1))
list2 = sorted(list1, key=lambda x:x[1])
'''
'''
pop = Population(100)
GA_op = pop.get_random_agent(3)
print(GA_op[0])
print(pop.pop[GA_op[0]])
'''
