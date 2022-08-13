# SnakeAI
Neural network uses genetic algorithm to learn how to play snake game.

How to use:
The snake.py file is used to train a snake AI. 
The neural network is then flatted and exported to a text file. 
An example neural network that has already been trained is in gareth.txt.
Copy and paste the contents of the text file into the snake.js file in the same format as for 'gareth'.
Open the snake.html file and watch the AI play a game of snake.

How does it work:
A population of random neural networks is generated.
They each play a game of snake and are given a score based off of how many apples were eaten.
The snakes' neural networks then crossover via the genetic algorithm.
The next generation of snakes repeats this process until a gareth is born.

Note:
You can tidy up some of my code if you want.
It would be handy to use js text file parsing to read the file rather than pasting it in to the source code.
