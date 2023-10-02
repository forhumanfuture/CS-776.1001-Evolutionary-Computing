import numpy as np

# Define the parameters
population_size = 50
num_generations = 100
crossover_prob = 0.7
mutation_prob = 0.001
num_variables = 5  # Number of variables in the optimization problem
variable_range = (-5.12, 5.12)  # Range for each variable

# Function to initialize a population with random binary strings
def initialize_population(population_size, num_variables):
    return np.random.uniform(variable_range[0], variable_range[1], size=(population_size, num_variables))

# Function to evaluate the Step Function (De Jong Function 3)
def evaluate_step_function(solution):
    return sum([x**2 for x in solution])

# Genetic algorithm main loop
population = initialize_population(population_size, num_variables)

for generation in range(num_generations):
    # Evaluate fitness of each individual
    fitness_values = [evaluate_step_function(individual) for individual in population]

    # Select parents for crossover (using tournament selection)
    parents = []
    for _ in range(population_size):
        tournament_indices = np.random.choice(population_size, size=2, replace=False)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        parent_index = tournament_indices[np.argmin(tournament_fitness)]
        parents.append(population[parent_index])

    # Perform crossover
    new_population = []
    for i in range(0, population_size, 2):
        if np.random.rand() < crossover_prob:
            crossover_point = np.random.randint(1, num_variables)
            child1 = np.concatenate((parents[i][:crossover_point], parents[i + 1][crossover_point:]))
            child2 = np.concatenate((parents[i + 1][:crossover_point], parents[i][crossover_point:]))
            new_population.extend([child1, child2])
        else:
            new_population.extend([parents[i], parents[i + 1]])

    # Perform mutation
    for i in range(population_size):
        for j in range(num_variables):
            if np.random.rand() < mutation_prob:
                new_population[i][j] = np.random.uniform(variable_range[0], variable_range[1])

    population = new_population

# Find the best solution
best_solution = population[np.argmin(fitness_values)]
best_fitness = evaluate_step_function(best_solution)

print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)




