#!/usr/bin/env python

from fantasybball.scrapers.get_htb_data import get_htb_data
import heapq
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

TOP_N = 6
heap = []

df = get_htb_data()

cols = ['FG%', 'FT%', '3PM',
        'PTS', 'TREB', 'AST',
        'STL', 'BLK', 'TO']

df = df[df['POS'] == 'C']

for i in range(len(cols) - 1):
    for j in range(i + 1, len(cols) - 1):
        x = np.array(df[cols[i]]).reshape(-1, 1)
        y = np.array(df[cols[j]]).reshape(-1, 1)

        reg = LinearRegression().fit(x, y)
        score = reg.score(x, y)

        if score > 0.25:
            plt.plot(x.flatten(), y.flatten(), 'o')
            plt.plot(x.flatten(), reg.predict(x).flatten())
            plt.title('{} vs {}, score: {}'.format(cols[i], cols[j], score))
            plt.show()


