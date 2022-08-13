import numpy as np
from random import random
from random import randint
from math import e
from time import time


def sigmoid(arr):
    for i in range(len(arr)):                                       # applies function to each element in array
        arr[i] = 1 / (1 + e**(-arr[i]))      # sigmoid is a common activation function

    return arr


class Layer:
    def __init__(self, size1, size2):                   # size1 is the number of neurons in the previous layer
        self.size = size2                               # size2 is the number of neurons in the layer
        self.activations = np.zeros(size2)
        if size1 != 0:
            self.weights = np.zeros((size2, size1))     # weights going into each node
            self.biases = np.zeros(size2)               # bias of each node
        else:
            self.weights = np.array([])                 # first layer has no weights or biases
            self.biases = np.array([])


class NeuralNetwork:
    def __init__(self, layout):
        self.size = len(layout)
        self.layers = []
        self.layers.append(Layer(0, layout[0]))
        for i in range(self.size - 1):
            self.layers.append(Layer(layout[i], layout[i + 1]))

    def randomise(self):                                    # give neural network random weights and biases
        for i in range(self.size - 1):
            layer = self.layers[i + 1]
            for j in range(layer.size):
                layer.biases[j] = 2 * random() - 1          # bias from -1 to 1
                for k in range(self.layers[i].size):
                    layer.weights[j, k] = 2 * random() - 1  # weight from -1 to 1

    def feed_forward(self):                                 # forward propagation
        for i in range(self.size - 1):
            layer1 = self.layers[i]
            layer2 = self.layers[i + 1]
            layer2.activations = sigmoid(np.dot(layer2.weights, layer1.activations) + layer2.biases)
            # weighted sum of activations

    def flatten(self):                               # outputs 2 x 1D arrays of the weights and biases
        fweights = np.array([])
        for i in range(self.size - 1):
            layer = self.layers[i + 1]
            for j in range(layer.size):
                fweights = np.append(fweights, layer.weights[j])

        fbiases = np.array([])
        for layer in self.layers:
            fbiases = np.append(fbiases, layer.biases)

        return [fweights, fbiases]

    def unflatten(self, genes):                 # takes 2 x 1D arrays and fills the weights and biases back in
        fweights = genes[0]
        fbiases = genes[1]
        counter1 = 0
        counter2 = 0
        for i in range(self.size - 1):
            layer = self.layers[i + 1]
            layer.biases = np.array(fbiases[counter1 : counter1 + layer.size])
            counter1 += layer.size
            for j in range(layer.size):
                layer.weights[j] = np.array(fweights[counter2 : counter2 + self.layers[i].size])
                counter2 += self.layers[i].size


class Snake:
    def __init__(self, topology):
        self.squares = [818, 819, 820, 821, 822]    # 0 is bottom left and works from left to right and top to bottom
        self.direction = 1                          # 0 North, 1 East, 2 South, 3 West
        self.nn = NeuralNetwork(topology)           # give the snake a brain
        self.dead = False                           # whether the snake has died yet
        self.grew = False                           # whether the snake just ate an apple and grew
        self.hunger = 500                           # number of steps before needing to eat an apple
        self.score = 0
        self.fitness = 0

    def turn(self, dir):
        # if -1 then left, if 0 then no change, if 1 then right
        self.direction += dir
        self.direction %= 4

    def move(self):
        moves = {0: 40, 1: 1, 2: -40, 3: -1}                            # 40 is the row size
        self.squares.append(self.squares[-1] + moves[self.direction])   # find the new square the head is on
        if self.grew:
            self.grew = False                                           # if snake grew then don't delete the tail
        else:
            self.squares.pop(0)                                         # delete the tail of the snake

    def onBoard(self, square):
        if square < 0 or square >= 1600:                    # snake is off board
            return 0
        if square % 40 == 39 and self.direction == 3:       # snake went left when on far left
            return 0
        if square % 40 == 0 and self.direction == 1:        # snake went right when on far right
            return 0

        return 1

    def checkDead(self):
        head = self.squares[-1]
        if not self.onBoard(head):                                  # snake went off the board
            return 1
        if len(self.squares) != len(set(self.squares)):             # snake hit itself
            return 1
        if self.hunger <= 0:                                        # snake hasn't eaten in too long
            return 1

        return 0

    def chooseMove(self, apple):
        distances = np.zeros(24)
        # 8 x 3 input neurons representing distance to wall, apple and body in 8 directions
        head = self.squares[-1]
        moves = [[40, 41, 1, -39, -40, -41, -1, 39], [1, -39, -40, -41, -1, 39, 40, 41],
                 [-40, -41, -1, 39, 40, 41, 1, -39], [-1, 39, 40, 41, 1, -39, -40, -41]][self.direction]

        # distance to walls
        bottom = head // 40
        top = 39 - bottom
        left = head % 40
        right = 39 - left
        distances[(self.direction * -2) % 8] = 1 / (top + 1)
        distances[(self.direction * -2 + 1) % 8] = 1 / (min(top, right) + 1)
        distances[(self.direction * -2 + 2) % 8] = 1 / (right + 1)
        distances[(self.direction * -2 + 3) % 8] = 1 / (min(bottom, right) + 1)
        distances[(self.direction * -2 + 4) % 8] = 1 / (bottom + 1)
        distances[(self.direction * -2 + 5) % 8] = 1 / (min(bottom, left) + 1)
        distances[(self.direction * -2 + 6) % 8] = 1 / (left + 1)
        distances[(self.direction * -2 + 7) % 8] = 1 / (min(top, left) + 1)

        # distance to apple
        y = (head // 40) - (apple // 40)
        x = (head % 40) - (apple % 40)
        if y == 0:
            if x > 0:
                distances[(6 - self.direction * 2) % 8 + 8] = 1 / x
            else:
                distances[(2 - self.direction * 2) % 8 + 8] = -1 / x
        elif x == 0:
            if y > 0:
                distances[(4 - self.direction * 2) % 8 + 8] = 1 / y
            else:
                distances[(self.direction * -2) % 8 + 8] = -1 / y
        elif abs(x) == abs(y):
            if y > 0:
                if x > 0:
                    distances[(5 - self.direction * 2) % 8 + 8] = 1 / y
                else:
                    distances[(7 - self.direction * 2) % 8 + 8] = 1 / y
            else:
                if x > 0:
                    distances[(3 - self.direction * 2) % 8 + 8] = -1 / y
                else:
                    distances[(1 - self.direction * 2) % 8 + 8] = -1 / y

        # distance to body
        for i in range(8):
            square = head + moves[i]
            distance = 1
            found = True
            while square not in self.squares:
                square += moves[i]
                if not self.onBoard(square):
                    found = False
                    break
                distance += 1

            if found:
                distances[i + 16] = 1 / distance

        # Note: 1/distance is used to account for when nothing is found, and also to normalise inputs

        self.nn.layers[0].activations = distances               # input distances to neural network
        self.nn.feed_forward()                                  # get neural network to decide how to move
        return np.argmin(self.nn.layers[-1].activations) - 1    # 0,1,2 gets mapped to -1,0,1 for turning

    def load(self):
        with open('gareth.txt', 'r') as f:
            genes = f.read().split('\n')
            fweights = list(map(float, genes[0].split(',')))
            fbiases = list(map(float, genes[1].split(',')))
            self.nn.unflatten([fweights, fbiases])


class Game:
    def __init__(self, snake):
        self.snake = snake
        self.placeApple()
        self.score = 0

    def placeApple(self):
        apple = randint(0, 1599)
        while apple in self.snake.squares:  # apple cannot be inside the snake
            apple = randint(0, 1599)

        self.apple = apple

    def checkApple(self):
        return self.snake.squares[-1] == self.apple  # check if head and apple are on the same square

    def play(self):  # simulates game until snake dies
        while not self.snake.dead:
            self.snake.turn(self.snake.chooseMove(self.apple))  # change snake direction
            self.snake.move()  # move snake 1 square in direction
            self.snake.hunger -= 1
            self.snake.dead = self.snake.checkDead()  # check if snake died
            if self.checkApple():  # if an apple is eaten
                self.score += 1  # total apples eaten -> how well snake performed
                self.snake.grew = True  # snake becomes longer
                self.snake.hunger = 500  # reset snake hunger
                self.placeApple()  # place a new apple on the board

        self.snake.score = self.score

    def display(self):
        board = ['0' for _ in range(1600)]
        for square in self.snake.squares:
            board[square] = '1'
        board[self.apple] = '2'

        text = ''
        for i in range(40):
            text = "".join(board[40 * i : 40 * (i + 1)]) + '\n' + text

        print(text)


class Population:
    def __init__(self, size, topology):
        self.size = size
        self.pool = []
        self.topology = topology
        self.generation = 1
        self.best = 0
        self.bestSnake = None
        self.bestSnakeFitness = 0

    def populate(self):
        for i in range(self.size):
            snake = Snake(self.topology)
            snake.nn.randomise()
            self.pool.append(snake)

    def fitness(self):                          # calculates fitness of every snake in the generation
        for snake in self.pool:
            f = snake.score + (500 - snake.hunger) / 5000
            snake.fitness = f ** 2
            if f > self.best:
                self.best = f                   # update best fitness if necessary
                if f > self.bestSnakeFitness:
                    self.bestSnake = snake
                    self.bestSnakeFitness = f

    def crossover(self, parent1, parent2):      # returns child snake of two parent snakes
        genes1 = parent1.nn.flatten()
        genes2 = parent2.nn.flatten()
        p1 = randint(0, len(genes1[0]) - 1)     # two point crossover
        p2 = randint(0, len(genes1[0]) - 1)
        new_weights = np.concatenate([genes1[0][:min(p1, p2)], genes2[0][min(p1, p2):max(p1, p2)], genes1[0][max(p1, p2):]])
        p1 = randint(0, len(genes1[1]) - 1)
        p2 = randint(0, len(genes1[1]) - 1)
        new_biases = np.concatenate([genes1[1][:min(p1, p2)], genes2[1][min(p1, p2):max(p1, p2)], genes1[1][max(p1, p2):]])
        snake = Snake(self.topology)
        snake.nn.unflatten(self.mutate([new_weights, new_biases]))      # new snake is mutated
        return snake

    def mutate(self, genes):
        probability = 0.01                      # mutation rate

        for i in range(len(genes[0])):
            if random() < probability:
                genes[0][i] = 2 * random() - 1  # weight from -1 to 1

        for j in range(len(genes[1])):
            if random() < probability:
                genes[1][j] = 2 * random() - 1  # bias from -1 to 1

        return genes

    def generate(self):
        self.generation += 1
        self.fitness()
        new = [self.bestSnake]              # best snake always lives on
        total = 0
        for snake in self.pool:
            total += snake.fitness

        for i in range(self.size - 1):
            r1 = random() * total
            r2 = random() * total

            for snake in self.pool:
                r1 -= snake.fitness
                if r1 <= 0:
                    parent1 = snake
                    break

            for snake in self.pool:
                r2 -= snake.fitness
                if r2 <= 0:
                    parent2 = snake
                    break

            new.append(self.crossover(parent1, parent2))

        self.pool = new

    def train(self, gens):
        self.populate()
        while self.generation <= gens:
            self.best = 0
            print('Generation: ' + str(self.generation))
            for snake in self.pool:
                game = Game(snake)
                game.play()

            self.generate()
            print('High score from this gen: ' + str(self.best))

        print('Highest score ever: ' + str(self.bestSnakeFitness))

        #self.export()

    def export(self):
        print('Exporting data...')
        data = self.bestSnake.nn.flatten()
        text = ''
        for weight in data[0]:
            text += str(weight) + ','

        text = text.rstrip(text[-1]) + '\n'

        for bias in data[1]:
            text += str(bias) + ','

        with open('gareth.txt', 'w') as f:
            f.write(text.rstrip(text[-1]))


if __name__ == '__main__':
    population = Population(1000, [24, 18, 18, 3])
    population.populate()
    population.train(20)