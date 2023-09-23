import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
class DistributionOfReturns:
    # use a discrete distribution supported on a list of atoms to approximate the return distribution.
    def __init__(self, numOfAtoms: int, probOfAtoms: torch.Tensor, gamma: float, device: torch.device):
        self.numOfAtoms = numOfAtoms
        self.device = device
        self.probOfAtoms = probOfAtoms.to(self.device)
        self.gamma = gamma
        self.step = (1.0 / (1 - self.gamma)) / (self.numOfAtoms - 1)
        self.atoms = torch.arange(start = 0.0, end = 1 / (1 - self.gamma) + self.step, step = self.step, device = self.device)
        assert self.probOfAtoms.shape == (self.numOfAtoms,)
        assert self.atoms.shape == (self.numOfAtoms,)
        # assert torch.sum(self.probOfBins).item() == 1.0
    
    def toDevice(self, device):
        self.device = device
        self.atoms = self.atoms.to(device)
        self.probOfAtoms = self.probOfAtoms.to(device)

    

    def resetProb(self, probOfAtoms: torch.Tensor):
        self.probOfAtoms = probOfAtoms.to(self.device)
        assert self.probOfAtoms.shape == (self.numOfAtoms,)
        # assert torch.sum(self.probOfBins).item() == 1.0

    def addDistribution(self, weight1, dist2, weight2):
        tmpProb = weight1 * self.probOfAtoms + weight2 * dist2.probOfAtoms
        return DistributionOfReturns(self.numOfAtoms, tmpProb, self.gamma, self.device)

    def resetProbWithSample(self, sample: torch.Tensor):
        sample = sample.to(self.device)
        probOfAtoms = torch.zeros_like(self.probOfAtoms, device = self.device)
        m = len(sample)
        # l = self.whichBin(sample)
        # u = l + 1
        # newProb = torch.zeros_like(probOfAtoms.repeat(m, 1))
        # tmpAtoms = self.atoms.repeat(m, 1)
        # newProb[torch.arange(m), l] += (sample - tmpAtoms[torch.arange(m), l]) / self.step / m
        # newProb[torch.arange(m), u] += (tmpAtoms[torch.arange(m), u] - sample) / self.step / m
        for x in tqdm(sample):
            l = self.whichBinSingle(x.item())
            u = l + 1
            # devide the probability of x between two neighboring atoms.
            probOfAtoms[l] += (x.item() - self.atoms[l]) / self.step / m
            probOfAtoms[u] += (self.atoms[u] - x.item()) / self.step / m
        # probOfAtoms = torch.sum(newProb, dim=0)
        self.probOfAtoms = probOfAtoms
    
    def resetProbWithWeightedSample(self, sample, prob):
        probOfAtoms = torch.zeros_like(self.probOfAtoms, device = self.device)
        n = len(sample)
        for i, x in enumerate(sample):
            l = self.whichBinSingle(x.item())
            u = l + 1
            # devide the probability of x between two neighboring atoms.
            probOfAtoms[l] += (x.item() - self.atoms[l]) / self.step * prob[i]
            probOfAtoms[u] += (self.atoms[u] - x.item()) / self.step * prob[i]
        self.probOfAtoms = probOfAtoms


    def whichBin(self, x):
        # determine which bin a tensor of returns belong, return a tensor of integer
        assert torch.all((x <= 1.0 / (1 - self.gamma) + 1e-4) & (x >= 0.0 - 1e-4)).item()
        return torch.floor(x / self.step).int()
    def whichBinSingle(self, x):
        # determine which bin a single return belong, return a integer
        assert (x <= 1.0 / (1 - self.gamma) + 1e-4) & (x >= 0.0 - 1e-4)
        return math.floor(x / self.step)
    def computeDensity(self):
        densityOfAtoms = self.probOfAtoms / self.step
        # correction
        densityOfAtoms[0] /= 2
        densityOfAtoms[-1] /= 2
        # move to CPU
        return densityOfAtoms.cpu()
    def plotDensity(self):
        densityOfAtoms = self.computeDensity()
        plt.stairs(densityOfAtoms.numpy(), np.arange(self.numOfAtoms + 1) * self.step, fill=True)
        plt.show()
    def computeCDF(self):
        cdfOfAtoms = torch.zeros_like(self.probOfAtoms)
        cdfOfAtoms[0] = self.probOfAtoms[0]
        for i in range(1, self.numOfAtoms):
            cdfOfAtoms[i] = cdfOfAtoms[i - 1] + self.probOfAtoms[i]
        return cdfOfAtoms.cpu()
    def plotCDF(self):
        cdfOfAtoms = self.computeCDF()
        plt.stairs(cdfOfAtoms.numpy(), np.arange(self.numOfAtoms + 1) * self.step, fill=False)
        plt.show()
    
    def computeVariance(self):
        # secondMoment and first Moment are floats
        secondMoment = torch.sum(torch.mul(torch.square(self.atoms), self.probOfAtoms)).cpu().item()
        firstMoment = torch.sum(torch.mul(self.atoms, self.probOfAtoms)).cpu().item()
        return secondMoment - firstMoment * firstMoment
    
    def computeIndexOfQuantile(self, p):
        Cdf = self.computeCDF()
        mask = (Cdf >= p) * 1
        tmpIndex = torch.argmax(mask)
        return tmpIndex.cpu().item()
    
    def computeQuantile(self, p):
        return self.atoms[self.computeIndexOfQuantile(p)].cpu().item()





        
            

        


        