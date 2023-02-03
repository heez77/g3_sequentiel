from numpy import random
from tqdm import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt

def generate_sample(env, Q, nb_sample, max_seq_length):
    """This function generates nb_sample trajectories using the matrix Q as probability matrix in the environment env. 
       A trajectory contains, for each row, the initial state, the chosen action and the related reward.

    Args:
        env (_type_): Maze environment
        Q (_type_): Porbability matrix
        nb_sample (_type_): Number of samples to generate
        max_seq_length (_type_): Max number of iteration for one trajectory.

    Returns:
        list: a list of trajectories.
    """
    sample = []
    States = env.cells
    for i in tqdm(range(nb_sample)):
        cell = env.reset(env.start_cell)
        s = States.index((cell[0][0], cell[0][1]))
        X = []
        a = random.choice([i for i in range(len(Q[s]))], p = Q[s])
        cell_, r, _ = env.step(a)
        s_ = States.index((cell_[0][0], cell_[0][1]))
        X.append(np.array([int(s), int(a), r]))
        t=0
        while r != -10 and t<max_seq_length:
            a = random.choice([i for i in range(len(Q[s_]))], p = Q[s_])
            s__ = s_
            cell_, r, _ = env.step(a)
            s_ = States.index((cell_[0][0], cell_[0][1]))
            X.append(np.array([int(s__), int(a), r]))
            t+=1
        X.append(np.array([int(s_), np.nan, np.nan]))
        sample.append(np.array(X))
    return sample

def score_sample(sample, alpha=0.5):
    """This function computes the score for each state in a trajectory based on equation 7 in the research paper. 

    Args:
        sample (list): a list of trajectories.
        alpha (float, optional): alpha parameter in equation 7. Defaults to 0.5.

    Returns:
        list: a list with the computed score for each state in each trajectories.
    """
    score = []
    for X in sample:
        score.append(np.array([np.sum([(alpha**i) * X[j+i,2] for i in range(len(X[j:-1]))]) for j in range(len(X)-1)]))
    return score

def calculate_good_trajectories(sample,Score, state, action=None):
    """This function calculates the number of trajectories where the state appears if action is None. 
       Otherwise calculates the number of trajectories where the state appears and where the 
       minimal score is for the given action.

    Args:
        sample (_type_): a list of trajectories.
        Score (_type_): a list with the computed score for each state in each trajectories.
        state (_type_): a given state.
        action (_type_, optional): a given action. Defaults to None.

    Returns:
        int: The number of trajectories
    """
    nb = 0
    if action is None:
        return len(sample)
    else:
        for X, S in zip(sample, Score):
            if X[:-1, 1][X[:-1, 0]==state][np.argmin(S[X[:-1, 0]==state])]==action:
                nb+=1
        return nb

def get_index(sample, gamma, Score, nb_states):
    """This function determines, for each state, the list of trajectories whose score is smaller than the gamma associated to the state

    Args:
        sample (list): a list of trajectories.
        gamma (np.array): The gamma list calculates in the CE algorithm.
        Score (list): a list with the computed score for each state in each trajectories.
        nb_states (int): Number of possible states.

    Returns:
        list: a list of sub-samples index for each state
    """
    index = []
    for state in range(nb_states):
        index_state = []
        for n, (X,S) in enumerate(zip(sample, Score)):
            if np.count_nonzero(S[X[:-1, 0]==state] <= gamma[state])>0:
                index_state.append(n)
        index.append(index_state)
    return index


def update_Q(sample, gamma, S, Q, nb_states):
    """This function estimate the new probability matrix using CE algorithm.
    Args:
        sample (list): a list of trajectories.
        gamma (np.array): The gamma list calculates in the CE algorithm.
        S (np.array): a list with the computed score for each state in each trajectories.
        Q (np.array): The probability matrix.
        nb_states (int): Number of possible states.

    Returns:
        np.array: The estimated Q matrix
    """
    Q_ = np.zeros(Q.shape)
    index = get_index(sample, gamma, S, nb_states)
    for s in range(len(Q)):
        if len(index[s])>0:
            sample_reduce =  list(map(lambda i: sample[i], index[s]))
            Score_reduce =  list(map(lambda i: S[i], index[s]))
            nb_in_s = calculate_good_trajectories(sample_reduce,Score_reduce, s)
            for a in range(len(Q[s])):
                nb_in_s_a = calculate_good_trajectories(sample_reduce,Score_reduce, s, a)
                if nb_in_s>0:
                    Q_[s, a] = nb_in_s_a / nb_in_s
        else:
            for a in range(len(Q[s])):
                Q_[s, a]=0
    return Q_  

def ce_resolution(sample, S, Q, nb_states, alpha=0.7, rho=0.03):
    """This function implements the algorithm 2.1 (CE algorithm) of the research paper

    Args:
        sample (list): a list of trajectories.
        S (np.array): a list with the computed score for each state in each trajectories.
        Q (np.array): The probability matrix.
        nb_states (int): Number of possible states.
        alpha (float, optional): alpha parameter in CE algorithm. Defaults to 0.7.
        rho (float, optional): rho parameter in CE algorithm. Defaults to 0.03.

    Returns:
        np.array: The gamma list calculates in the CE algorithm.
        np.array: The updated probability matrix.
    """
    gamma = np.zeros(nb_states)
    for j in range(nb_states):
        S_ = np.array([S[n][i]  for n, X in enumerate(sample) for i in range(len(X)-1) if X[i, 0]==j])
        if len(S_)>0:
            gamma[j] = np.percentile(np.sort(S_), (1-rho)*100)
        else:
            gamma[j] = -np.inf

    Q_ = update_Q(sample, gamma, S, Q, nb_states)
    for i in range(len(Q)):
        if np.sum(Q_[i])==1:
            Q[i] = alpha*Q_[i] + (1-alpha)*Q[i]
    return gamma, Q

dx_ = {0:-0.25,1:0.25,2:0,3:0}
dy_ = {0:0,1:0,2:-0.25,3:0.25}

plt.rcParams["figure.figsize"] = (10,10)

def print_grid_path(env, path):
    """This function display the trajectory for a given path.

    Args:
        env : Maze env
        path (np.array): The trajectory to be displayed
    """
    plt.clf()
    plt.imshow(np.abs(env.maze-1), cmap="gray", vmin=0, vmax=1)
    plt.grid(True)
    cells = env.cells
    for i in range(len(path)-1):
        s = int(path[i,0])
        a = int(path[i,1])
        plt.arrow(cells[s][0], cells[s][1], dx_[a], dy_[a], width = 0.05)
    plt.show()
    
    
def q_learning_cross_entropy(env, nb_sample=100,  T=50, d=3,  max_seq_length=100):
    """This function performs the CE alogithm and displays the best trajectory at each sample generation. 
       The algorithm stops if the gamma list has not changed d times.

    Args:
        env : Maze env
        nb_sample (int, optional): Number of sample to use in the algorithme research. Defaults to 100.
        T (int, optional): Maximum of iterations. Defaults to 50.
        d (int, optional): _description_. Defaults to 3.
        max_seq_length (int, optional): early stop parameter. Defaults to 100.

    Returns:
        list: a list with the last trajectories generated
        Q : the final Q matrix
    """
    S = len(env.cells)
    A = len(env.actions)
    Q = np.full((S, A), 1/A)
    sample = generate_sample(env, Q, nb_sample, max_seq_length)
    Score = score_sample(sample)
    gamma, Q = ce_resolution(sample, Score, Q, S)
    t=1
    n= 0
    print("Best score is : ", min([np.sum(X[:-1, 2]) for X in sample]))
    print_grid_path(env, sample[np.argmin([np.sum(X[:-1, 2]) for X in sample])])
    while t<T and n!=d:
        sample = generate_sample(env, Q, nb_sample, max_seq_length)
        Score = score_sample(sample)
        gamma_, Q = ce_resolution(sample, Score, Q, S)
        print("Best score is : ",min([np.sum(X[:-1, 2]) for X in sample]))
        if np.allclose(gamma,gamma_):
            n+=1
        else:
            n=0
        gamma = gamma_
        t+=1
        print_grid_path(env, sample[np.argmin([np.sum(X[:-1, 2]) for X in sample])])
    print_grid_path(env, sample[np.argmin([np.sum(X[:-1, 2]) for X in sample])])
    print('total step :', t)
    return sample, Q 