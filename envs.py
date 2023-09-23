import os
import time
import math
from tqdm import tqdm
import torch
import numpy as np
from torch import Tensor
from torch.linalg import solve
from TruncatedNormal import TruncatedNormal
from utils import torchVectorizeChoice

class TabularMDP:
    def __init__(self, name: str, sizeOfStateSpace: int, sizeOfActionSpace: int, gamma: float,
                 P: Tensor, rewardMean: Tensor, rewardStd: Tensor, device: torch.device,
                 *args, **kwargs):
        # Reward are set as truncated Gaussian on [0,1].
        self.name = name
        self.device = device
        self.sizeS = sizeOfStateSpace
        self.sizeA = sizeOfActionSpace
        self.stateSpace = range(sizeOfStateSpace)
        self.actionSpace = range(sizeOfActionSpace)
        self.gamma = gamma
        self.P = torch.as_tensor(P, dtype=torch.float32, device=self.device)  # (S, A, S)
        self.rewardMean = torch.as_tensor(rewardMean, dtype=torch.float32, device=self.device)  # (S, A)
        self.rewardStd = torch.as_tensor(rewardStd, dtype=torch.float32, device=self.device) #(S, A)
        # self.initialState = initialState  # Initial distribution
        # P, R, mu, lb_Y and ub_Y should be valid
        assert self.P.shape == (self.sizeS, self.sizeA, self.sizeS)
        assert torch.allclose(torch.sum(self.P, dim=2), torch.ones((self.sizeS, self.sizeA), device=self.device))
        assert self.rewardMean.shape == (self.sizeS, self.sizeA)
        assert self.rewardStd.shape == (self.sizeS, self.sizeA)
        assert torch.all((self.rewardMean < 1.0) & (self.rewardMean > 0.0)).item()
        assert torch.all(self.rewardStd > 0.0).item()
    
    def toDevice(self, device):
        self.device = device
        self.P = self.P.to(device)
        self.rewardMean = self.rewardMean.to(device)
        self.rewardStd = self.rewardStd.to(device)
    # Check whether an input policy pi is a valid policy
    def checkPi(self, pi: Tensor):
        assert pi.shape == (self.sizeS, self.sizeA)
        assert torch.allclose(torch.sum(pi, dim=1), torch.ones((self.sizeS), device=self.device))
        return
    
    # (N, S, A)
    def getReward(self, numOfSamples):
        reward = TruncatedNormal(self.rewardMean, self.rewardStd, torch.zeros_like(self.rewardMean), torch.ones_like(self.rewardMean))
        return reward.rsample(torch.Size([numOfSamples]))
    


    # do not sample from the Reward.
    def getSingleReward(self, s, a):
        assert s in self.stateSpace and s in self.actionSpace
        gaussian = torch.distributions.Normal(loc=self.rewardMean[s, a], scale=self.rewardStd[s,a])
        while(True):
            r = gaussian.sample()
            if (r <= 1 and r > 0):
                return r


    def Ppi(self, pi: Tensor) -> Tensor:
        self.checkPi(pi)
        piAxis = pi[:, :, None]
        return torch.sum(self.P * piAxis, dim=1)
    
    def Rpi(self, pi: Tensor) -> Tensor:
        self.checkPi(pi)
        return torch.sum(pi * self.rewardMean, dim=1)
    
    def Vpi(self, pi: Tensor) -> Tensor:
        Rpi = self.Rpi(pi)
        Ppi = self.Ppi(pi)
        return solve(torch.eye(self.sizeS, device=self.device) - self.gamma * Ppi, Rpi)
    
    
    # Q value function, (S, A)
    def Qpi(self, pi: Tensor) -> Tensor:
        Vpi = self.Vpi(pi)
        return self.rewardMean + self.gamma * torch.inner(self.P, Vpi)
    
    def PhatGenerativeModel(self, m) -> Tensor:
        SAArray = m * torch.ones((self.sizeS, self.sizeA), dtype=torch.long, device=self.device)
        SASArray = torch.zeros((self.sizeS, self.sizeA, self.sizeS), device=self.device)
        for s in range(self.sizeS):
            for a in range(self.sizeA):
                SASArray[s, a, :] = torch.tensor(np.random.multinomial(n=SAArray[s, a].item(), pvals=self.P[s, a].cpu().numpy()), device=self.device)
        return SASArray / torch.maximum(SAArray[:, :, None], torch.as_tensor([1], device=SASArray.device))
    
    def generateEmpiricalMDP(self, m):
        Phat = self.PhatGenerativeModel(m)
        return TabularMDP("MDP estimated from" + str(m) + "samples.", self.sizeS, self.sizeA, self.gamma, Phat,
                        self.rewardMean, self.rewardStd, self.device)
    
    def monteCarloSampleOfCumulativeReward(self, pi: Tensor, numOfTraj: int, lenOfTraj: int,
                                          initialState: int) -> Tensor:
        self.checkPi(pi)

        currentStates = torch.tensor(initialState, device=self.device).repeat(numOfTraj)
            # Sample a0 from s0 and pi
        currentActions = torchVectorizeChoice(pTensor=pi[currentStates, :], device=self.device)
        distributionOfRewards = TruncatedNormal(loc=self.rewardMean[currentStates, currentActions],
                                                scale=self.rewardStd[currentStates, currentActions], a=torch.zeros_like(currentStates), b=torch.ones_like(currentStates))
        cumulativeReward = distributionOfRewards.rsample()
        # Simulate trajectories with vectorization
        for t in tqdm(range(lenOfTraj-1)):
            # Sample s_{t+1} from s_t and a_t
            currentStates = torchVectorizeChoice(pTensor=self.P[currentStates, currentActions], device=self.device)
            # Sample a_{t+1} from s_{t+1} and pi
            currentActions = torchVectorizeChoice(pTensor=pi[currentStates, :], device=self.device)
            distributionOfRewards = TruncatedNormal(loc=self.rewardMean[currentStates, currentActions],
                                                scale=self.rewardStd[currentStates, currentActions], a=torch.zeros_like(currentStates), b=torch.ones_like(currentStates))
            cumulativeReward += (distributionOfRewards.rsample() * (self.gamma ** (t + 1.)))
        return cumulativeReward

class RandomMDP(TabularMDP):
    def __init__(self, name: str, sizeOfStateSpace: int, sizeOfActionSpace: int, gamma: float, rewardScale: float,
                device: torch.device, *args, **kwargs):
        
        P = np.random.uniform(0.1, 0.9, size=(sizeOfStateSpace, sizeOfActionSpace, sizeOfStateSpace))
        P = P / (np.sum(P, axis=2)[:, :, np.newaxis])
        P = torch.as_tensor(P, device=device)

        rewardMean = np.random.uniform(low=0, high=1, size=(sizeOfStateSpace, sizeOfActionSpace))
        rewardMean = torch.as_tensor(rewardMean, device=device)
        rewardStd = rewardScale * torch.ones_like(rewardMean)

        super(RandomMDP, self).__init__(name=name, sizeOfStateSpace=sizeOfStateSpace, sizeOfActionSpace=sizeOfActionSpace, gamma=gamma, P=P,
                                                         rewardMean=rewardMean, rewardStd=rewardStd, device=device)
        return

    