# Computational Intelligence Exam Project - Quarto
Repository for Computational Intelligence exam project.

## How to use the Agent in tournament
To develop my agent I used the Montecarlo Tree Search algorithm.
To be able to code this algorithm I had to add to the Quarto class some methods that were necessary.

As requsted I created a subclass called `CustomQuarto`, so that is needed to play the game.

The agent class is called `MCTSAgent` and just like the `RandomPlayer` needs as input the Quarto game object.
In the `MCTSAgent` constructor could also be specified some parameters as:
- `num_rounds`: value that specifies a depth limit to the algorithm, meaning number of random games played at each decision step 
- `c`: value useful for tuning the amount of exploration in the Upper Confidence Bound calculation

Both these parameters are set by default to values that, in my testing, I found to be good enough against the random agent, setting the constraints to be able to produce a move in a reasonable time.
