from DistributionOfReturns import DistributionOfReturns
import torch
from tqdm import tqdm
from TruncatedNormal import TruncatedNormal
def pushForward(eta, mdp, s, a):
    rsa = TruncatedNormal(mdp.rewardMean[s, a], mdp.rewardStd[s, a], torch.zeros_like(mdp.rewardMean[s, a]), torch.ones_like(mdp.rewardStd[s, a]))
    step = eta.step
    numOfAtoms = eta.numOfAtoms
    probOfAtoms = eta.probOfAtoms
    atoms = eta.atoms
    gamma = mdp.gamma
    lowerInterval = torch.t(atoms.repeat(numOfAtoms, 1)) - step / 2 - gamma * atoms.repeat(numOfAtoms, 1)
    upperInterval = torch.t(atoms.repeat(numOfAtoms, 1)) + step / 2 - gamma * atoms.repeat(numOfAtoms, 1)
    weightedMat = rsa.cdf(upperInterval) - rsa.cdf(lowerInterval)
    return torch.matmul(weightedMat, probOfAtoms)

def batchPushForward(probsOfAtoms, atoms, step, mdp, s, a):
    rsa = TruncatedNormal(mdp.rewardMean[s, a], mdp.rewardStd[s, a], torch.zeros_like(mdp.rewardMean[s, a]), torch.ones_like(mdp.rewardStd[s, a]))
    # probOfAtoms: batchSize * numOfAtoms
    numOfAtoms = probsOfAtoms.size(dim = 1)
    gamma = mdp.gamma
    lowerInterval = torch.t(atoms.repeat(numOfAtoms, 1)) - step / 2 - gamma * atoms.repeat(numOfAtoms, 1)
    upperInterval = torch.t(atoms.repeat(numOfAtoms, 1)) + step / 2 - gamma * atoms.repeat(numOfAtoms, 1)
    weightedMat = rsa.cdf(upperInterval) - rsa.cdf(lowerInterval)
    return torch.matmul(weightedMat, probsOfAtoms.T).T
     
    
def pushForwardBySample(eta, r): 
    # eta is a DistribuutionOfReturns; r is a tensor of size (m, ), which is the sampled reward.
        m = len(r)
        newAtoms = eta.gamma * eta.atoms
        newAtoms = newAtoms.repeat(m, 1) + torch.t(r.repeat(eta.numOfAtoms, 1)) # (m, numOfAtoms)
        # print(newAtoms.shape)
        newProb = torch.zeros_like(newAtoms, device = eta.device)
        tmpAtoms = eta.atoms.repeat(m, 1)
        tmpProb = eta.probOfAtoms.repeat(m, 1)
        for i in range(newProb.shape[1]):
            l = torch.maximum(eta.whichBin(newAtoms[:, i]), torch.zeros_like(newAtoms[:, i], dtype=torch.int8))
            u = torch.minimum(l + int(1), int(eta.numOfAtoms - 1) * torch.ones_like(l))
            newProb[torch.arange(m), l] += (newAtoms[:, i] - tmpAtoms[torch.arange(m), l]) / eta.step * tmpProb[torch.arange(m), i]
            newProb[torch.arange(m), u] += (tmpAtoms[torch.arange(m), u] - newAtoms[:, i]) / eta.step * tmpProb[torch.arange(m), i]
        newProb = torch.mean(newProb, 0)
        return newProb

def distributionalDP(etas, M, pi):
    # input: a list of DistributionOfReturns; MDP M; policy pi; reward rs.
    numOfAtoms = etas[0].numOfAtoms
    gamma = etas[0].gamma
    stateSpace = M.stateSpace
    actionSpace = M.actionSpace
    device = etas[0].device
    newEtas = [DistributionOfReturns(numOfAtoms, torch.ones(numOfAtoms)/(numOfAtoms + 0.0), gamma, device) for _ in stateSpace]
    for s in stateSpace:
        tmpProb = torch.zeros(numOfAtoms, device=device)
        for a in actionSpace:
            for sPrime in stateSpace:
                tmpProb += M.P[s, a, sPrime] * pi[s, a] * pushForward(etas[sPrime], M, s, a)
        newEtas[s].resetProb(tmpProb)
    return newEtas

def batchDistributionalDP(listOfProbsOfAtoms, atoms, step, M, pi):
    # input: a list of probsOfAtoms; MDP M; policy pi; reward rs.
    numOfAtoms = listOfProbsOfAtoms[0].size(dim = 1)
    gamma = M.gamma
    stateSpace = M.stateSpace
    actionSpace = M.actionSpace
    device = M.device
    newListOfProbs = [torch.zeros_like(probsOfAtoms) for probsOfAtoms in listOfProbsOfAtoms]
    for s in stateSpace:
        for a in actionSpace:
            for sPrime in stateSpace:
                newListOfProbs[s] += M.P[s, a, sPrime] * pi[s, a] * batchPushForward(listOfProbsOfAtoms[sPrime], atoms, step, M, s, a)
    return newListOfProbs