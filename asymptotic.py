import os
import torch
import pickle
from envs import RandomMDP
from utils import randomPolicy, computeKS, computeTV
from DistributionOfReturns import DistributionOfReturns
from distributionalDP import distributionalDP
# from nonAsymptotic import getTrueEtas
import ot
from tqdm import tqdm
import numpy as np
import argparse
import math
from RandomElement import RandomElement

nS = 5 # Number of states.
nA = 2 # Number of actions.
# initialS = 0 # Initial state.
rewardScale = 0.1 # std of the reward distribution (truncated Gaussian).
nAtoms = 1000 # number of atoms in our catogorial representation
nRepeat = 1000 # repeat multiple times and take the average

parser = argparse.ArgumentParser()
parser.add_argument("--n", help = "Size of dataset.", default = 100, type = int)
parser.add_argument("--gamma", help = "discount factor of the MDP", default = 0.9, type = float)
parser.add_argument("--device", help = "which CUDA device", default = "cuda:0", type = str)
parser.add_argument("--inferenceType", help = "which type of inferences", default = "KS", type = str)
parser.add_argument("--whichQuantile", help = "only valid if inferenceType is quantile", default = 0.9, type = float)

args = parser.parse_args()
n = args.n
gamma = args.gamma
whichQuantile = args.whichQuantile
device = torch.device(args.device)
inferenceType = args.inferenceType
nIte = int(-6 * math.log(10) / math.log(gamma)) # how many iterations of distributional DP.

def getEstimates(mdp, n, pi, device):
        # return the final etas and an Array of W distances between estimatedEta and trueEta
    estimatedMDP = mdp.generateEmpiricalMDP(n)
    gamma = estimatedMDP.gamma
    estimatedEtas = [DistributionOfReturns(nAtoms, torch.ones(nAtoms) / (nAtoms + 0.0), gamma, device) for s in estimatedMDP.stateSpace]
    iteBar = tqdm(range(nIte))
    for t in iteBar:
        iteBar.set_description(f"Estimaing eta... DP iteration {t}")
        estimatedEtas = distributionalDP(estimatedEtas, estimatedMDP, pi)
    return estimatedMDP, estimatedEtas
    
def getTrueEtas(mdp, pi, device):
    gamma = mdp.gamma
    trueEtas = [DistributionOfReturns(nAtoms, torch.ones(nAtoms) / (nAtoms + 0.0), gamma, device) for s in mdp.stateSpace]
    iteBar = tqdm(range(nIte))
    for t in iteBar:
        iteBar.set_description(f"DP iteration {t}")
        trueEtas = distributionalDP(trueEtas, mdp, pi)
    return trueEtas

def computeKSCoverage(mdp, trueEtas, n, pi, nInferences, device):
    repeatBar = tqdm(range(nInferences))
    sumOfSuccess = 0
    CIRadius = torch.zeros(nInferences)
    for i in repeatBar:
        estimatedMDP, estimatedEtas = getEstimates(mdp, n, pi, device)
        randomElement = RandomElement(estimatedMDP, pi, estimatedEtas, device)
        repeatBar.set_description(f"Inference of KS distance: the {i} th try.")
        tmp = randomElement.sampleKSDistance(1000, 0)
        upperConfidence = torch.quantile(tmp, 0.95, interpolation='linear').item() / math.sqrt(n)
        CIRadius[i] = upperConfidence
        if (computeKS(trueEtas[0].computeCDF(), estimatedEtas[0].computeCDF()).cpu().item() <= upperConfidence):
            sumOfSuccess += 1
    return sumOfSuccess * 1.0 / nInferences, CIRadius

def computeWassersteinCoverage(mdp, trueEtas, n, pi, nInferences, device):
    repeatBar = tqdm(range(nInferences))
    sumOfSuccess = 0
    CIRadius = torch.zeros(nInferences)
    for i in repeatBar:
        estimatedMDP, estimatedEtas = getEstimates(mdp, n, pi, device)
        randomElement = RandomElement(estimatedMDP, pi, estimatedEtas, device)
        repeatBar.set_description(f"Inference of Wasserstein distance: the {i} th try.")
        tmp = randomElement.sampleWassersteinDistance(1000, 0)
        upperConfidence = torch.quantile(tmp, 0.95, interpolation='linear').item() / math.sqrt(n)
        CIRadius[i] = upperConfidence
        WDistance = ot.wasserstein_1d(trueEtas[0].atoms, estimatedEtas[0].atoms, trueEtas[0].probOfAtoms, estimatedEtas[0].probOfAtoms).cpu().item()
        if (WDistance <= upperConfidence):
            sumOfSuccess += 1
    return sumOfSuccess * 1.0 / nInferences, CIRadius

def computeTVCoverage(mdp, trueEtas, n, pi, nInferences, device):
    repeatBar = tqdm(range(nInferences))
    sumOfSuccess = 0
    CIRadius = torch.zeros(nInferences)
    for i in repeatBar:
        estimatedMDP, estimatedEtas = getEstimates(mdp, n, pi, device)
        randomElement = RandomElement(estimatedMDP, pi, estimatedEtas, device)
        repeatBar.set_description(f"Inference of TV distance: the {i} th try.")
        tmp = randomElement.sampleTVDistance(1000, 0)
        upperConfidence = torch.quantile(tmp, 0.95, interpolation='linear').item() / math.sqrt(n)
        CIRadius[i] = upperConfidence
        TVDistance = computeTV(trueEtas[0].probOfAtoms, estimatedEtas[0].probOfAtoms)
        if (TVDistance <= upperConfidence):
            sumOfSuccess += 1
    return sumOfSuccess * 1.0 / nInferences, CIRadius

def computeQuantileCoverage(mdp, trueEtas, n, pi, nInferences, device, p):
    repeatBar = tqdm(range(nInferences))
    sumOfSuccess = 0
    CIRadius = torch.zeros(nInferences)
    for i in repeatBar:
        estimatedMDP, estimatedEtas = getEstimates(mdp, n, pi, device)
        randomElement = RandomElement(estimatedMDP, pi, estimatedEtas, device)
        repeatBar.set_description(f"Inference of {int(100 * p)}th quantile: the {i} th try.")
        tmp = randomElement.sampleQuantile(1000, 0, p)
        upperConfidence = torch.quantile(tmp, 0.975, interpolation='linear').item() / math.sqrt(n)
        lowerConfidence = torch.quantile(tmp, 0.025, interpolation='linear').item() / math.sqrt(n)
        CIRadius[i] = (upperConfidence - lowerConfidence) / 2
        sampleQuantile = estimatedEtas[0].computeQuantile(p)
        trueQuantile = trueEtas[0].computeQuantile(p)
        # print(sampleQuantile - trueQuantile)
        # print(f"[{lowerConfidence}, {upperConfidence}]")
        if (sampleQuantile - trueQuantile <= upperConfidence and sampleQuantile - trueQuantile >= lowerConfidence):
            sumOfSuccess += 1
    return sumOfSuccess * 1.0 / nInferences, CIRadius

def computeVarianceCoverage(mdp, trueEtas, n, pi, nInferences, device):
    repeatBar = tqdm(range(nInferences))
    sumOfSuccess = 0
    CIRadius = torch.zeros(nInferences)
    for i in repeatBar:
        estimatedMDP, estimatedEtas = getEstimates(mdp, n, pi, device)
        randomElement = RandomElement(estimatedMDP, pi, estimatedEtas, device)
        repeatBar.set_description(f"Inference of variance: the {i} th try.")
        tmp = randomElement.sampleVarianceOfReturn(1000, 0)
        upperConfidence = torch.quantile(tmp, 0.975, interpolation='linear').item() / math.sqrt(n)
        lowerConfidence = torch.quantile(tmp, 0.025, interpolation='linear').item() / math.sqrt(n)
        CIRadius[i] = (upperConfidence - lowerConfidence) / 2 
        sampleVariance = estimatedEtas[0].computeVariance()
        trueVariance = trueEtas[0].computeVariance()
        # print(trueVariance - sampleVariance)
        # print(f"[{lowerConfidence}, {upperConfidence}]")
        if (trueVariance <= sampleVariance + upperConfidence and trueVariance >= sampleVariance + lowerConfidence):
            sumOfSuccess += 1
    return sumOfSuccess * 1.0 / nInferences, CIRadius


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
    if inferenceType == "KS":
        coverageRate, CIRadius = computeKSCoverage(mdp, trueEtas, n, pi, nRepeat, device)
        print("Coverage Rate" + str(coverageRate))
        with open("./datas/asymptoticResult.txt", "a") as f:
            f.write(f"Coverage Rate of KS inference when n = {n}: {coverageRate}\n")
            f.write(f"Averaged CI radius of KS inference when n = {n}: {torch.mean(CIRadius).item()}\n")
            f.write(f"Std of CI radius of KS inference when n = {n}: {torch.std(CIRadius).item()}\n")
            f.write("\n")
    if inferenceType == "TV":
        coverageRate, CIRadius = computeTVCoverage(mdp, trueEtas, n, pi, nRepeat, device)
        print("Coverage Rate" + str(coverageRate))
        with open("./datas/asymptoticResult.txt", "a") as f:
            f.write(f"Coverage Rate of TV inference when n = {n}: {coverageRate}\n")
            f.write(f"Averaged CI radius of TV inference when n = {n}: {torch.mean(CIRadius).item()}\n")
            f.write(f"Std of CI radius of TV inference when n = {n}: {torch.std(CIRadius).item()}\n")
            f.write("\n")
    elif inferenceType == "Wasserstein":
        coverageRate, CIRadius = computeWassersteinCoverage(mdp, trueEtas, n, pi, nRepeat, device)
        print("Coverage Rate" + str(coverageRate))
        with open("./datas/asymptoticResult.txt", "a") as f:
            f.write(f"Coverage Rate of Wasserstein inference when n = {n}: {coverageRate}\n")
            f.write(f"Averaged CI radius of Wasserstein inference when n = {n}: {torch.mean(CIRadius).item()}\n")
            f.write(f"Std of CI radius of Wasserstein inference when n = {n}: {torch.std(CIRadius).item()}\n")
            f.write("\n")
    elif inferenceType == "quantile":
        coverageRate, CIRadius = computeQuantileCoverage(mdp, trueEtas, n, pi, nRepeat, device, whichQuantile)
        print("Coverage Rate" + str(coverageRate))
        with open("./datas/asymptoticResult.txt", "a") as f:
            f.write(f"Coverage Rate of {int(whichQuantile * 100)}th quantile inference when n = {n}: {coverageRate}\n")
            f.write(f"Averaged CI radius of {int(whichQuantile * 100)}th quantile inference when n = {n}: {torch.mean(CIRadius).item()}\n")
            f.write(f"Std of CI radius of {int(whichQuantile * 100)}th quantile inference when n = {n}: {torch.std(CIRadius).item()}\n")
            f.write("\n")
    elif inferenceType == "variance":
        coverageRate, CIRadius = computeVarianceCoverage(mdp, trueEtas, n, pi, nRepeat, device)
        print("Coverage Rate" + str(coverageRate))
        with open("./datas/asymptoticResult.txt", "a") as f:
            f.write(f"Coverage Rate of variance inference when n = {n}: {coverageRate}\n")
            f.write(f"Averaged CI radius of variance inference when n = {n}: {torch.mean(CIRadius).item()}\n")
            f.write(f"Std of CI radius of variance inference when n = {n}: {torch.std(CIRadius).item()}\n")
            f.write("\n")
    else:
        print("Error. No such options.")
    return

if __name__ == '__main__':
    main()

