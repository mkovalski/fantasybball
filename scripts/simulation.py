#!/usr/bin/env python

from fantasybball.scrapers.get_htb_data import get_htb_data
import numpy as np

CATEGORIES = ['FG%', 'FT%', '3PM', 'PTS',
              'TREB', 'AST', 'STL', 'BLK', 'TO']

class Player:
    def __init__(self, num, strategy = 'random'):
        self.num = num
        self.strategy = strategy
        self.indices = []
        self.players = []

    def choose(self, df):
        # Why can't I choose a fucking view here, this is terrible
        avail = df[df.TAKEN == 0]
        idx = np.random.choice(avail.index)
        df.iloc[idx, df.columns.get_loc('TAKEN')] = self.num
        self.players.append(df.iloc[idx].PLAYER)
        self.indices.append(idx)

    def get_score(self, df):
        columns = [df.columns.get_loc(x) for x in CATEGORIES]
        players = df.iloc[self.indices, columns]
        res = np.sum(players, axis = 0)
        res['TO'] = res['TO'] * -1

        return res

def rank_players(players, df):
    scores = []
    for i in range(len(players)):
        score = 0
        for j in range(len(players)):
            if i == j:
                continue

            p1_score = players[i].get_score(df)
            p2_score = players[j].get_score(df)
            score += int(
                np.sum(p1_score > p2_score) > 4)

        scores.append(score)
    return scores

SPOTS = 10
PLAYERS = 8

df = get_htb_data()

# Add some columns
df.insert(2, 'TAKEN', np.zeros(len(df)).astype(np.int))

players = [Player(num = i + 1, strategy = 'random') for i in range(PLAYERS)]

assert(SPOTS*PLAYERS <= len(df))

for i in range(SPOTS):
    for player in players:
        player.choose(df)

score = rank_players(players, df)

for i in range(len(players)):
    print("{}: {}".format(score[i], players[i].players))
