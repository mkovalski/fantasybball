#!/usr/bin/env python

from fantasybball.scrapers.get_htb_data import get_htb_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = get_htb_data()

cols = ['FG%', 'FT%', '3PM',
        'PTS', 'TREB', 'AST',
        'STL', 'BLK', 'TO']

for col in cols:
    data = df[col]
    plt.plot(data, np.zeros_like(data), 'o')
    plt.title(col)
    plt.show()


