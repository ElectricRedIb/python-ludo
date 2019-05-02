import numpy as np
import random
from Agent import Agent
from tempfile import TemporaryFile
from os import path


class Population:
    size_Of_Pop = 100
    file = '/pop_pool/simple_pop'
    script_dir = path.dirname(__file__)
    generation = 1
    amount_of_genes = 4
    mu = 0
    sigma = 10

    def __init__(self):
        None

    def init_pop(self):
        self.pop = [np.random.normal(self.mu, self.sigma, self.amount_of_genes) for _ in range(self.size_Of_Pop)]

    def fitness(self, fitness, index_of_chromosome):
        None

    def breed(self, sampled_pop, fitness):
        p1 = np.random.choice(sampled_pop, replace=False, p=fitness)
        p2 = np.random.choice(sampled_pop, replace=False, p=fitness)
        [child_a, child_b] = self.crossover(parent_a=p1, parent_b=p2)
        self.pop = self.execute(new_agents=[child_a, child_b], sampled_pop=sampled_pop, fitness=fitness)
        self.generation = self.generation + 1

    def crossover(self, parent_a, parent_b):
        mask = np.random.rand(self.amount_of_genes) < 0.5
        child_a = parent_a * mask + parent_b * n.invert(mask)
        child_b = parent_b * mask + parent_a * n.invert(mask)
        child_a = self.mutation(child_a)
        child_b = self.mutation(child_b)
        return [child_a, child_b]

    def mutation(self, child, mutation_rate=0.1):
        mask = np.random.rand(self.amount_of_genes) < mutation_rate
        mutation = [i for i, x in enumerate(mask) if x]
        for i in mutation:
            child[i] = np.random.normal(child[i], 1, self.amount_of_genes)[0]
        return child

    def execute(self, new_agents, sampled_pop, fitness):
        execution = [1 - i for i in fitness]

        execution[:] = [i / sum(execution) for i in execution]
        relace_agent_idxs = np.random.choice(np.arange(len(sampled_pop)), len(new_agents), replace=False, p=execution)
        for i in range(len(replace_agent_idxs)):
            sampled_pop[replace_agent_idxs[i]] = new_agents[i]
        return sampled_pop

    def save_pop(self):
        np.savez(self.script_dir + self.file + str(self.generation) + '.pool', ch=self.pop)

    def tournament(self, total_fighters, total_fights):
        fighters = np.random.choice(self.pop, replace=False, 6)
