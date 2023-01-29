# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
import argparse
import quarto
import random

class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.CustomQuarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)

def main():
    game = quarto.CustomQuarto()
    game.set_players((RandomPlayer(game), quarto.MCTSAgent(game, num_rounds=1000, c=10.0)))
    winner = game.run()
    logging.warning(f"main: Winner: player {winner}")

def games_test(games: int = 100):
    wincount = 0
    drawcount = 0
    losecount = 0
    as_player_one = 0
    for g in range(games):
        game = quarto.CustomQuarto()
        players = (RandomPlayer(game), quarto.MCTSAgent(game, num_rounds=1000, c=10.0))
        agent_turn = random.randint(0, 1)
        as_player_one += agent_turn
        if agent_turn == 1:
            game.set_players((players[0], players[1]))
        else:
            game.set_players((players[1], players[0]))
        print(f"Playing game number {g} - Agent is player #{agent_turn} - ", end='')
        winner = game.run_noprint()
        if winner == agent_turn: wincount += 1
        if winner == (1 - agent_turn): losecount += 1
        if winner == -1: drawcount += 1
        print(f"winner is player #{winner}")
    print(f"Against random player:\n- Winrate: {wincount/games*100:.2f}%\n- Drawrate: {drawcount/games*100:.2f}%\n- Loserate: {losecount/games*100:.2f}%")
    print(f"Agent started as player #1 {as_player_one/games*100:.2f}% of the games, player #0 as {1 - as_player_one/games*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    #main()
    games_test()