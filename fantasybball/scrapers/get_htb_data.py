#!/usr/bin/env python

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

URL = 'https://hashtagbasketball.com/fantasy-basketball-rankings'

def remove_empty(df):
    df = df[df['R#'] != 'R#']
    return df

def clean_fg(df):
    clean_percentage(df, 'FG%')

def clean_ft(df):
    clean_percentage(df, 'FT%')

def clean_percentage(df, key):
    df.loc[:, key] = df.loc[:, key].apply(lambda x: x.split(' ')[0])
    df[key] = df[key].astype(np.float32)

def clean_data(df):
    df = remove_empty(df)

    clean_fg(df)
    clean_ft(df)

    for key in ['3PM', 'PTS', 'TREB', 'AST', 'STL', 'BLK', 'TO', 'TOTAL']:
        df[key] = df[key].astype(np.float32)

    df['R#'] = df['R#'].astype(np.int)

    df.index = range(len(df))

    return df

def get_htb_data():
    '''Grab data from hashtag basketball. Uses simple layout from site'''
    # TODO: Add customization for things like > 200 players in table view
    page = requests.get(URL)
    page.raise_for_status()

    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find_all('table')[2]
    df = pd.read_html(str(table))

    df = df[0]
    # Clean up the columns a bit
    df = clean_data(df)

    return df

if __name__ == '__main__':
    get_htb_data()

