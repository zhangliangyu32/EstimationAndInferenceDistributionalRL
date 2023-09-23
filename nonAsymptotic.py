# hyper-parameters
import os
import torch
import pickle
from envs import RandomMDP
from utils import randomPolicy, computeKS, computeTV
from DistributionOfReturns import DistributionOfReturns
from distributionalDP import distributionalDP
import ot
from tqdm import tqdm
import numpy as np
import argparse
import math

# MCSteps = 10000000 # Use MC to compute the ground truth.
# MCHorizon = 1000
nS = 5 # Number of states.
nA = 2 # Number of actions.
# initialS = 0 # Initial state.
rewardScale = 0.1 # std of the reward distribution (truncated Gaussian).
nAtoms = 1000 # number of atoms in our catogorial representation
nRepeat = 100 # repeat multiple times and take the average

# ns = [10, 100, 1000, 10000] # Try datasets with different sizes.
# p = [1, 2, 3, 4] # Try different Wp metric.
# gammas = [0.7, 0.8, 0.9, 0.97, 0.98] # Try different Gammas

# device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument("--n", help = "Size of dataset.", default = 100, type = int)
parser.add_argument("--gamma", help = "discount factor of the MDP", default = 0.9, type = float)
parser.add_argument("--device", help = "which CUDA device", default = "cuda:0", type = str)

args = parser.parse_args()
n = args.n
gamma = args.gamma
device = torch.device(args.device)
nIte = int(-6 * math.log(10) / math.log(gamma)) # number of iterations of distributional DP.
def getEstimatedEtaErr(mdp, n, pi, trueEtas, device):
        # return the final etas and an Array of W distances between estimatedEta and trueEta
    wDistances = np.zeros((nIte, mdp.sizeS))
    KSDistances = np.zeros((nIte, mdp.sizeS))
    TVDistances = np.zeros((nIte, mdp.sizeS))
    estimatedMDP = mdp.generateEmpiricalMDP(n)
    gamma = estimatedMDP.gamma
    estimatedEtas = [DistributionOfReturns(nAtoms, torch.ones(nAtoms) / (nAtoms + 0.0), gamma, device) for s in estimatedMDP.stateSpace]
    iteBar = tqdm(range(nIte))
    for t in iteBar:
        iteBar.set_description(f"DP iteration {t}")
        estimatedEtas = distributionalDP(estimatedEtas, estimatedMDP, pi)
        for s in estimatedMDP.stateSpace:
            wDistances[t, s] = ot.wasserstein_1d(trueEtas[s].atoms, estimatedEtas[s].atoms, trueEtas[s].probOfAtoms, estimatedEtas[s].probOfAtoms).cpu().item()
            KSDistances[t, s] = computeKS(trueEtas[s].computeCDF(), estimatedEtas[s].computeCDF()).cpu().item()
            TVDistances[t, s] = computeTV(trueEtas[s].probOfAtoms, estimatedEtas[s].probOfAtoms)
    return wDistances, KSDistances, TVDistances, estimatedEtas

def getTrueEtas(mdp, pi, device):
    gamma = mdp.gamma
    trueEtas = [DistributionOfReturns(nAtoms, torch.ones(nAtoms) / (nAtoms + 0.0), gamma, device) for s in mdp.stateSpace]
    iteBar = tqdm(range(nIte))
    for t in iteBar:
        iteBar.set_description(f"DP iteration {t}")
        trueEtas = distributionalDP(trueEtas, mdp, pi)
    return trueEtas

def main():
    if os.path.exists(f'./datas/MDPGamma={gamma}.pkl'):
        # Load the saved MDP.
        with open(f'./datas/MDPGamma={gamma}.pkl', 'rb') as file:
            # MDP is a RandomMDP with gamma as the discount factor.
            mdp = pickle.load(file)
        mdp.toDevice(device)
        print(f"MDP loaded on {device}.")
    else:
        mdp = RandomMDP("Random MDP", nS, nA, gamma, rewardScale, device)
        with open(f'./datas/MDPGamma={gamma}.pkl', 'wb') as file:
            pickle.dump(mdp, file)
    
    if os.path.exists('./datas/policy.pkl'):
        # Load the saved policy.
        with open('./datas/policy.pkl', 'rb') as file:
            pi = pickle.load(file)
        pi = pi.to(device)
        print(f"Policy loaded on {device}.")
    else:
        pi = randomPolicy(nS, nA, device)
        with open('./datas/policy.pkl', 'wb') as file:
            pickle.dump(pi, file)
    
    if os.path.exists(f'./datas/groundTruthGamma={gamma}.pkl'):
        # Load the saved ground truth.
        with open(f'./datas/groundTruthGamma={gamma}.pkl', 'rb') as file:
            # True etas is a list of Distribution of returns.
            trueEtas = pickle.load(file)
        for eta in trueEtas:
            eta.toDevice(device)
        print(f"Ground truth loaded on {device}.")
    else:
        # Use distributional DP to find a ground truth.
        trueEtas = getTrueEtas(mdp, pi, device)
        with open(f'./datas/groundTruthGamma={gamma}.pkl', 'wb') as file:
            pickle.dump(trueEtas, file)

    if os.path.exists(f'./datas/averageWDistancesn={n}Gamma={gamma}.pkl') and os.path.exists(f'./datas/averageKSDistancesn={n}Gamma={gamma}.pkl')\
        and os.path.exists(f'./datas/averageTVDistancesn={n}Gamma={gamma}.pkl'):
        # Load the saved W distance data.
        with open(f'./datas/averageWDistancesn={n}Gamma={gamma}.pkl', 'rb') as file:
            # averageWDistances is a numpy array.
            averageWDistances = pickle.load(file)
        print("W distance data loaded.")
        # Load the saved KS distance data.
        with open(f'./datas/averageKSDistancesn={n}Gamma={gamma}.pkl', 'rb') as file:
            # averageKSDistances is a dictionary numpy array.
            averageKSDistances = pickle.load(file)
        print("KS distance data loaded.")
        with open(f'./datas/averageTVDistancesn={n}Gamma={gamma}.pkl', 'rb') as file:
            # averageKSDistances is a dictionary numpy array.
            averageTVDistances = pickle.load(file)
        print("TV distance data loaded.")
    else:
        averageWDistances = np.zeros((nIte, mdp.sizeS))
        averageKSDistances = np.zeros((nIte, mdp.sizeS))
        averageTVDistances = np.zeros((nIte, mdp.sizeS))
        repeatBar = tqdm(range(nRepeat))
        tmpWDistances = np.zeros((nRepeat, nIte, mdp.sizeS))
        tmpKSDistances = np.zeros((nRepeat, nIte, mdp.sizeS))
        tmpTVDistances = np.zeros((nRepeat, nIte, mdp.sizeS))
        for i in repeatBar:
            repeatBar.set_description(f"the {i}th try")
            # only store the final estimated etas.
            tmpWDistances[i, :, :], tmpKSDistances[i, :, :], tmpTVDistances[i, :, :], estimatedEtas = getEstimatedEtaErr(mdp, n, pi, trueEtas, device)
        averageWDistances = np.mean(tmpWDistances, axis=0)
        averageKSDistances = np.mean(tmpKSDistances, axis=0)
        averageTVDistances = np.mean(tmpKSDistances, axis=0)
        with open(f'./datas/averageWDistancesn={n}Gamma={gamma}.pkl', 'wb') as file:
            pickle.dump(averageWDistances, file)
        with open(f'./datas/averageKSDistancesn={n}Gamma={gamma}.pkl', 'wb') as file:
            pickle.dump(averageKSDistances, file)
        with open(f'./datas/averageTVDistancesn={n}Gamma={gamma}.pkl', 'wb') as file:
            pickle.dump(averageTVDistances, file)
        with open(f'./datas/estimatedEtasn={n}Gamma={gamma}.pkl', 'wb') as file:
            pickle.dump(estimatedEtas, file)
    return

if __name__ == '__main__':
    main()

