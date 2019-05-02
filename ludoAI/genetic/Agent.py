import numpy as np


class Agent:
    amount_of_genes = 4
    min = 0
    max = 1

    def __init__(self):
        self.chromosome = np.random.uniform(self.min, self.max, self.amount_of_genes)
        self.fitness = 0

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome
