from DegenerateNormal import DegenerateNormal
import math
from DistributionOfReturns import DistributionOfReturns
from tqdm import tqdm
from distributionalDP import distributionalDP, pushForward, batchPushForward, batchDistributionalDP
import torch
from utils import computeCDF
class RandomElement:
    def __init__(self, mdp, pi, etas, device):
        self.device = device
        self.mdp = mdp
        self.P = mdp.P
        self.pi = pi
        self.etas = etas

        self.gaussians = []
        for s in self.mdp.stateSpace:
            self.gaussians.append([])
            for a in self.mdp.actionSpace:
                loc = torch.zeros(self.mdp.sizeS, device = self.device)
                cov = torch.diag(self.P[s, a, :]) - torch.matmul(self.P[s, a, :].unsqueeze(1), self.P[s, a, :].unsqueeze(1).T)
                self.gaussians[s].append(DegenerateNormal(loc, cov, self.device))
        
        self.pushedProbs = torch.zeros((self.mdp.sizeS, self.mdp.sizeA, self.mdp.sizeS, self.etas[0].numOfAtoms), device = self.device)
        for s in self.mdp.stateSpace:
            for a in self.mdp.actionSpace:
                for sPrime in self.mdp.stateSpace:
                    self.pushedProbs[s, a, sPrime, :] = pushForward(self.etas[sPrime], self.mdp, s, a)

    
    def neumannSeries(self, etas):
        gamma = self.mdp.gamma
        nIte = 30
        numOfAtoms = etas[0].numOfAtoms
        iteBar = tqdm(range(nIte))
        tempEtas = etas
        for t in iteBar:
            iteBar.set_description(f"Computing Neumann series... DP iteration {t}")
            tempEtas = distributionalDP(tempEtas, self.mdp, self.pi)
            tempEtas = [tempEtas[i].addDistribution(1, etas[i], 1) for i in range(len(tempEtas))]
            # correction
            for eta in tempEtas:
                eta.probOfAtoms = eta.probOfAtoms - torch.mean(eta.probOfAtoms)
        return tempEtas
    
    def batchNeumannSeries(self, listOfProbsOfAtoms):
        gamma = self.mdp.gamma
        atoms = self.etas[0].atoms
        step = self.etas[0].step
        numOfAtoms = self.etas[0].numOfAtoms
        nIte = int(-6 * math.log(10) / math.log(gamma)) # number of iterations of distributional DP.
        iteBar = tqdm(range(nIte))
        tempListOfProbs = listOfProbsOfAtoms
        for t in iteBar:
            iteBar.set_description(f"Computing Neumann series... DP iteration {t}")
            tempListOfProbs = batchDistributionalDP(tempListOfProbs, atoms, step, self.mdp, self.pi)
            tempListOfProbs = [tempListOfProbs[i] + listOfProbsOfAtoms[i] for i in range(len(tempListOfProbs))]
            # correction
            for i in range(len(tempListOfProbs)):
                tempListOfProbs[i] = tempListOfProbs[i] - torch.mean(tempListOfProbs[i], 1, True)
        return tempListOfProbs
    
    def sampleKSDistance(self, numOfSamples, initialState):
        sizeS = self.mdp.sizeS
        sizeA = self.mdp.sizeA
        stateSpace = self.mdp.stateSpace
        actionSpace = self.mdp.actionSpace
        Z = torch.zeros(numOfSamples, sizeS, sizeA, sizeS, device = self.device)
        for s in stateSpace:
            for a in actionSpace:
                Z[:, s, a, :] = self.gaussians[s][a].sample(numOfSamples)
                # correction
                Z[:, s, a, :] -= torch.mean(Z[:, s, a, :], 1, True)
        numOfAtoms = self.etas[0].numOfAtoms
        listOfProbs = [torch.zeros(numOfSamples, numOfAtoms, device=self.device) for s in stateSpace]
        for s in stateSpace:
            for a in actionSpace:
                for sPrime in stateSpace:
                    listOfProbs[s] = listOfProbs[s] + self.pi[s, a] * Z[:, s, a, sPrime].unsqueeze(1) * self.pushedProbs[s, a, sPrime].unsqueeze(0)
            # correction
            listOfProbs[s] = listOfProbs[s] - torch.mean(listOfProbs[s], 1, True)
        listOfProbs = self.batchNeumannSeries(listOfProbs)
        return torch.max(torch.abs(computeCDF(listOfProbs[initialState])), dim = 1)[0].cpu()
    
    def sampleTVDistance(self, numOfSamples, initialState):
        sizeS = self.mdp.sizeS
        sizeA = self.mdp.sizeA
        stateSpace = self.mdp.stateSpace
        actionSpace = self.mdp.actionSpace
        Z = torch.zeros(numOfSamples, sizeS, sizeA, sizeS, device = self.device)
        for s in stateSpace:
            for a in actionSpace:
                Z[:, s, a, :] = self.gaussians[s][a].sample(numOfSamples)
                # correction
                Z[:, s, a, :] -= torch.mean(Z[:, s, a, :], 1, True)
        numOfAtoms = self.etas[0].numOfAtoms
        listOfProbs = [torch.zeros(numOfSamples, numOfAtoms, device=self.device) for s in stateSpace]
        for s in stateSpace:
            for a in actionSpace:
                for sPrime in stateSpace:
                    listOfProbs[s] = listOfProbs[s] + self.pi[s, a] * Z[:, s, a, sPrime].unsqueeze(1) * self.pushedProbs[s, a, sPrime].unsqueeze(0)
            # correction
            listOfProbs[s] = listOfProbs[s] - torch.mean(listOfProbs[s], 1, True)
        listOfProbs = self.batchNeumannSeries(listOfProbs)
        return 0.5 * torch.sum(torch.abs(listOfProbs[initialState]), dim = 1).cpu()
    
    def sampleWassersteinDistance(self, numOfSamples, initialState):
        sizeS = self.mdp.sizeS
        sizeA = self.mdp.sizeA
        stateSpace = self.mdp.stateSpace
        actionSpace = self.mdp.actionSpace
        Z = torch.zeros(numOfSamples, sizeS, sizeA, sizeS, device = self.device)
        for s in stateSpace:
            for a in actionSpace:
                Z[:, s, a, :] = self.gaussians[s][a].sample(numOfSamples)
                # correction
                Z[:, s, a, :] -= torch.mean(Z[:, s, a, :], 1, True)
        numOfAtoms = self.etas[0].numOfAtoms
        atoms = self.etas[0].atoms
        step = self.etas[0].step
        listOfProbs = [torch.zeros(numOfSamples, numOfAtoms, device=self.device) for s in stateSpace]
        for s in stateSpace:
            for a in actionSpace:
                for sPrime in stateSpace:
                    listOfProbs[s] = listOfProbs[s] + self.pi[s, a] * Z[:, s, a, sPrime].unsqueeze(1) * self.pushedProbs[s, a, sPrime].unsqueeze(0)
            # correction
            listOfProbs[s] = listOfProbs[s] - torch.mean(listOfProbs[s], 1, True)
        listOfProbs = self.batchNeumannSeries(listOfProbs)
        return step * torch.sum(torch.abs(computeCDF(listOfProbs[initialState])), dim = 1).cpu()
    
    def sampleVarianceOfReturn(self, numOfSamples, initialState):
        sizeS = self.mdp.sizeS
        sizeA = self.mdp.sizeA
        stateSpace = self.mdp.stateSpace
        actionSpace = self.mdp.actionSpace
        Z = torch.zeros(numOfSamples, sizeS, sizeA, sizeS, device = self.device)
        for s in stateSpace:
            for a in actionSpace:
                Z[:, s, a, :] = self.gaussians[s][a].sample(numOfSamples)
                # correction
                Z[:, s, a, :] -= torch.mean(Z[:, s, a, :], 1, True)
        numOfAtoms = self.etas[0].numOfAtoms
        atoms = self.etas[0].atoms
        step = self.etas[0].step
        listOfProbs = [torch.zeros(numOfSamples, numOfAtoms, device=self.device) for s in stateSpace]
        for s in stateSpace:
            for a in actionSpace:
                for sPrime in stateSpace:
                    listOfProbs[s] = listOfProbs[s] + self.pi[s, a] * Z[:, s, a, sPrime].unsqueeze(1) * self.pushedProbs[s, a, sPrime].unsqueeze(0)
            # correction
            listOfProbs[s] = listOfProbs[s] - torch.mean(listOfProbs[s], 1, True)
        listOfProbs = self.batchNeumannSeries(listOfProbs)
        secondMoment = torch.sum(torch.mul(torch.square(atoms), listOfProbs[initialState]), dim = 1).cpu()
        firstMomentH = torch.sum(torch.mul(atoms, listOfProbs[initialState]), dim = 1).cpu()
        firstMomentEta = torch.sum(torch.mul(atoms, self.etas[initialState].probOfAtoms)).cpu().item()
        return secondMoment - 2 * firstMomentEta * firstMomentH
    
    def sampleQuantile(self, numOfSamples, initialState, p):
        #sample the pth quantile
        sizeS = self.mdp.sizeS
        sizeA = self.mdp.sizeA
        stateSpace = self.mdp.stateSpace
        actionSpace = self.mdp.actionSpace
        Z = torch.zeros(numOfSamples, sizeS, sizeA, sizeS, device = self.device)
        for s in stateSpace:
            for a in actionSpace:
                Z[:, s, a, :] = self.gaussians[s][a].sample(numOfSamples)
                # correction
                Z[:, s, a, :] -= torch.mean(Z[:, s, a, :], 1, True)
        numOfAtoms = self.etas[0].numOfAtoms
        atoms = self.etas[0].atoms
        step = self.etas[0].step
        listOfProbs = [torch.zeros(numOfSamples, numOfAtoms, device=self.device) for s in stateSpace]
        for s in stateSpace:
            for a in actionSpace:
                for sPrime in stateSpace:
                    listOfProbs[s] = listOfProbs[s] + self.pi[s, a] * Z[:, s, a, sPrime].unsqueeze(1) * self.pushedProbs[s, a, sPrime].unsqueeze(0)
            # correction
            listOfProbs[s] = listOfProbs[s] - torch.mean(listOfProbs[s], 1, True)
        listOfProbs = self.batchNeumannSeries(listOfProbs)
        indexOfPhiPEta = self.etas[initialState].computeIndexOfQuantile(p)
        Cdf = computeCDF(listOfProbs[initialState]).cpu()
        return 0 - torch.div(Cdf[:, indexOfPhiPEta], self.etas[initialState].probOfAtoms[indexOfPhiPEta].cpu()) * step



        
        