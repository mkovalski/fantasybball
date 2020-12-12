#!/usr/bin/env python

from fantasybball.scrapers import get_htb_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FantasyBBallEnv():
    COLS = ['FG%', 'FT%', '3PM',
            'PTS', 'TREB', 'AST',
            'STL', 'BLK', 'TO']
    def __init__(self,
                 num_opponents = 10,
                 num_picks = 10,
                 shuffle = False):
        '''Create a FantasyBBall Gym environmet for RL training

        Args:
            num_opponents (int, optional): Number of opponents
            num_picks (int, optional): Number of slots available for each player
            shuffle (bool, optional): Shuffle the board so that players are in
                random order in the state

        '''
        self.num_opponents = num_opponents
        self.num_picks = num_picks
        self.shuffle = shuffle

        # Grab the data from the htb website
        self.df = get_htb_data()

        # Initial states
        self.init_df_state, self.scalers = self.__normalize_dataframe()
        self.init_state = np.array(self.init_df_state)
        self.init_state.flags.writeable = False

        # To be set by reset() function
        self.player_num = None
        self.curr_round = None
        self.curr_state = None
        self.index_to_df = []

    def __normalize_dataframe(self):
        '''Normalize the dataframe and get the scaler for each'''
        new_df = self.df[self.COLS].copy()
        scalers = {}

        for col in self.COLS:
            scaler = StandardScaler()
            new_df[col] = scaler.fit_transform(np.array(new_df[col]).reshape(-1, 1))
            scalers[col] = scaler

        new_df.insert(0, 'TAKEN', np.zeros(len(new_df)).astype(np.int))
        return new_df, scalers

    def __random_opponent_action(self, state, player):
        '''Apply a random action for the opponents

        Args:
            state (np.array): The current state
            player (int): The marker for the player

        '''
        indices = np.where(state[:, 0] == 0)[0]
        idx = np.random.choice(indices)
        state[idx, 0] = player

    def reset(self):
        '''Reset the environment'''
        self.player_num = np.random.randint(1, self.num_opponents + 1)
        self.curr_round = 0

        self.curr_state = self.init_state.copy()

        # Shuffle up the board
        if self.shuffle:
            indices = np.arange(0, self.curr_state.shape[0])
            np.random.shuffle(indices)
            self.curr_state = self.curr_state[indices, :]
            self.index_to_df = indices

        for i in range(1, self.player_num):
            self.__random_opponent_action(state = self.curr_state, player = i)

        return np.copy(self.curr_state)

    def sample(self, state):
        # Filter based on available moves
        indices = np.where(state[:, 0] == 0)[0]
        idx = np.random.choice(indices)
        action = np.zeros(state.shape[0])
        action[idx] = 1
        return action

    def clean_action(self, state, action):
        indices = np.where(state[:, 0] != 0)[0]
        action[indices] = float('-inf')
        return action

    def calculate_reward(self, state):
        '''
        Basic reward structure, based on expected states determine
        how many head to head matchups one would win

        '''
        player_map = {}

        # TODO: Calculate these things during each action instead of all at the end
        for i in range(state.shape[0]):
            owner = state[i, 0]
            if owner != 0:
                if owner not in player_map:
                    player_map[owner] = []
                player_map[owner].append(i)

        me = player_map.pop(-1)
        columns = [self.df.columns.get_loc(x) for x in self.COLS]
        my_scores = self.df.iloc[me, columns]

        # Sum up the total score and invert turnovers
        my_scores = np.sum(my_scores, axis = 0)
        my_scores['TO'] *= -1

        won = 0
        for key, value in player_map.items():
            opp_scores = self.df.iloc[value, columns]
            opp_scores = np.sum(opp_scores, axis = 0)
            opp_scores['TO'] *= -1

            num_winners = np.sum(my_scores > opp_scores)

            won += (num_winners > (my_scores.count() // 2))

        return won / len(player_map)

    def step(self, action):
        action = self.clean_action(self.curr_state, action)
        idx = np.argmax(action)

        self.curr_state[idx, 0] = -1 # -1 marks us for now

        # Next player to end of players
        for i in range(self.player_num + 1, self.num_opponents + 1):
            self.__random_opponent_action(state = self.curr_state, player = i)

        # Check if we are done
        self.curr_round += 1
        if self.curr_round == self.num_picks:
            reward = self.calculate_reward(self.curr_state)
            return np.copy(self.curr_state), reward, True, 'Completed'

        # Loop around again
        for i in range(1, self.player_num):
            self.__random_opponent_action(state = self.curr_state, player = i)

        return np.copy(self.curr_state), 0, False, 'Still grindin'

if __name__ == '__main__':
    env = FantasyBBallEnv()
    state = env.reset()

    done = False

    while not done:
        action = env.sample(state)
        next_state, reward, done, info = env.step(action)

