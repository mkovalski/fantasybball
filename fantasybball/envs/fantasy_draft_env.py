#!/usr/bin/env python

from collections import OrderedDict
from fantasybball.scrapers import get_htb_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FantasyBBallEnv():
    COLS = ['FG%', 'FT%', '3PM',
            'PTS', 'TREB', 'AST',
            'STL', 'BLK', 'TO']
    POS = OrderedDict(PG = 1, SG = 2, SF = 3, PF = 4, C = 5)

    TAKEN_INDEX = 0

    def __init__(self,
                 num_opponents = 10,
                 num_picks = 10,
                 shuffle = False,
                 flatten_state = False,
                 max_centers = 4,
                 opponent_strategy = 'random'):
        '''Create a FantasyBBall Gym environmet for RL training

        Args:
            num_opponents (int, optional): Number of opponents
            num_picks (int, optional): Number of slots available for each player
            shuffle (bool, optional): Shuffle the board so that players are in
                random order in the state
            flatten_state (bool, optional): Flatten the state for input to
                linear model

        '''
        self.opponent_strategy_functions = dict(random = self.__random_opponent_action,
                                                ranking = self.__ranking_opponent_action)

        assert(opponent_strategy in self.opponent_strategy_functions.keys())

        self.num_opponents = num_opponents
        self.num_picks = num_picks
        self.shuffle = shuffle
        self.flatten_state = flatten_state
        self.max_centers = max_centers
        self.opponent_strategy = opponent_strategy

        self.opponent_function = self.opponent_strategy_functions[self.opponent_strategy]

        # Grab the data from the htb website
        self.df = get_htb_data()

        # Initial states
        self.init_df_state, self.scalers = self.__normalize_dataframe()
        self.init_state = np.array(self.init_df_state)
        self.init_state.flags.writeable = False

        # Get center indices for filtering moves
        self.center_indices = self.get_center_indices()

        # To be set by reset() function
        self.player_num = None
        self.curr_round = None
        self.curr_state = None
        self.centers_chosen = None

        if not self.shuffle:
            self.index_to_df = np.arange(0, self.init_state.shape[0]).astype(np.int)
        else:
            self.index_to_df = []

        self.df_to_idx = None

    def get_center_indices(self):
        centers = self.df.POS.apply(lambda x: x.split(','))

        indices = []
        # TODO: Optimize
        for i in range(len(centers)):
            if 'C' in centers.iloc[i]:
                indices.append(i)

        return set(indices)

    def __normalize_dataframe(self):
        '''Normalize the dataframe and get the scaler for each'''
        new_df = self.df[self.COLS].copy()
        scalers = {}

        # Standard scalars
        for col in self.COLS:
            scaler = StandardScaler()
            new_df[col] = scaler.fit_transform(np.array(new_df[col]).reshape(-1, 1))
            scalers[col] = scaler

        # One hot encode the positions available
        all_positions = np.array(self.df['POS'].apply(lambda x: x.split(',')))

        # Add all the columns
        for pos in self.POS.keys():
            new_df[pos] = np.zeros(len(new_df)).astype(np.uint8)

        for i in range(len(all_positions)):
            for pos in all_positions[i]:
                new_df.iloc[i, new_df.columns.get_loc(pos)] = 1

        new_df.insert(self.TAKEN_INDEX, 'TAKEN', np.zeros(len(new_df)).astype(np.int))
        return new_df, scalers

    def __random_opponent_action(self, state, player):
        '''Apply a random action for the opponents

        Args:
            state (np.array): The current state
            player (int): The marker for the player

        '''
        # Find only open slots to sample
        indices = np.where(state[:, self.TAKEN_INDEX] == 0)[0]
        idx = np.random.choice(indices)
        state[idx, 0] = player

    def __ranking_opponent_action(self, state, player):
        # Select the highest ranked available player 95% of the time

        if np.random.random() > 0.95:
            self.__random_opponent_action(state, player)
            return

        indices = np.where(state[:, self.TAKEN_INDEX] == 0)[0]
        df_indices = indices
        if self.shuffle:
            df_indices = self.index_to_df[indices]

        # Choose highest available ranking
        rankings = self.df.iloc[df_indices]['R#']
        idx = np.argmin(rankings)
        state[df_indices[idx], 0] = player


    def get_state_shape(self):
        if self.flatten_state:
            return int(np.prod(self.init_state.shape))
        return self.init_state.shape

    def get_output_size(self):
        return self.init_state.shape[0]

    def reset(self):
        '''Reset the environment'''
        self.player_num = np.random.randint(1, self.num_opponents + 1)
        self.curr_round = 0
        self.centers_chosen = 0

        self.curr_state = self.init_state.copy()

        # Shuffle up the board
        if self.shuffle:
            indices = np.arange(0, self.curr_state.shape[0]).astype(np.int)
            np.random.shuffle(indices)
            self.curr_state = self.curr_state[indices, :]
            self.index_to_df = indices

        for i in range(1, self.player_num):
            self.opponent_function(state = self.curr_state, player = i)

        if self.flatten_state:
            return np.copy(self.curr_state.flatten())
        return np.copy(self.curr_state)

    def reshape_state(self, state, batch = False):
        if not batch:
            return state.reshape(self.init_state.shape)

        batch_size = state.shape[0]
        return state.reshape((batch_size, *self.init_state.shape))

    def sample(self, state):
        if self.flatten_state:
            state = self.reshape_state(state)
        # Filter based on available moves
        indices = np.where(state[:, self.TAKEN_INDEX] == 0)[0]
        idx = np.random.choice(indices)
        action = np.zeros(state.shape[0])
        action[idx] = 1
        return action

    def __clean_action(self, state, action):
        # Internal version of cleaning up an action
        removed_indices = np.where(state[:, self.TAKEN_INDEX] != 0)[0]
        if self.centers_chosen >= self.max_centers:
            if self.shuffle:
                raise NotImplementedError

            removed_indices = np.array(list(set(removed_indices).union(self.center_indices)))

        action[removed_indices] = float('-inf')

    def clean_action(self, state, action):
        if self.flatten_state:
            state = self.reshape_state(state, batch = True)

        indices = np.where(state[:, :, self.TAKEN_INDEX] != 0)
        action[indices] = float('-inf')

    def calculate_reward(self, state):
        '''
        Basic reward structure, based on expected states determine
        how many head to head matchups one would win

        '''
        player_map = {}

        if self.flatten_state:
            state = self.reshape_state(state)

        # TODO: Calculate these things during each action instead of all at the end
        for i in range(state.shape[0]):
            owner = state[i, 0]
            if owner != 0:
                if owner not in player_map:
                    player_map[owner] = []
                player_map[owner].append(i)

        # Get original order of indices
        if self.shuffle:
            for key, value in player_map.items():
                player_map[key] = self.index_to_df[np.array(value)]

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
        self.__clean_action(self.curr_state, action)
        idx = np.argmax(action)

        self.curr_state[idx, 0] = -1 # -1 marks us for now

        # Update if we chose a center
        df_idx = idx
        if self.shuffle:
            df_idx = self.index_to_df[idx]

        if df_idx in self.center_indices:
            self.centers_chosen += 1

        # Next player to end of players
        for i in range(self.player_num + 1, self.num_opponents + 1):
            self.opponent_function(state = self.curr_state, player = i)

        # Check if we are done
        self.curr_round += 1
        if self.curr_round == self.num_picks:
            reward = self.calculate_reward(self.curr_state)

            ret_state = np.copy(self.curr_state)
            if self.flatten_state:
                ret_state = ret_state.flatten()

            return ret_state, reward, True, 'Completed'

        # Loop around again
        for i in range(1, self.player_num):
            self.opponent_function(state = self.curr_state, player = i)

        ret_state = np.copy(self.curr_state)
        if self.flatten_state:
            ret_state = ret_state.flatten()

        return ret_state, 0, False, 'Still grindin'

    def print_action(self, action, move_num):
        idx = np.argmax(action)
        if self.shuffle:
            idx = self.index_to_df[idx]

        player_selected = self.df.iloc[idx].PLAYER
        pick_num = (move_num * self.num_opponents) + self.player_num
        print(" - With the {} pick, selected {}".format(pick_num, player_selected))

if __name__ == '__main__':
    env = FantasyBBallEnv(shuffle = True)
    state = env.reset()

    done = False

    while not done:
        action = env.sample(state)
        next_state, reward, done, info = env.step(action)

