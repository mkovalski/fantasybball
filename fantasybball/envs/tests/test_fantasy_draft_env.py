#!/usr/bin/env python

from fantasybball.env import FantasyBBallEnv
import numpy as np

def test_reset():
    env = FantasyBBallEnv()
    state = env.reset()

    # Check that the main player has not been set
    chosen = state[:, 0]
    assert(not(any(chosen == -1)))

    # Check that each player before has been set
    my_player = env.player_num

    for i in range(1, my_player):
        indices = np.where(chosen == i)[0]
        assert(len(indices) == 1)

    # Make sure internal state and provided state are copies
    state += 1
    assert(not(np.any(state == env.curr_state)))





