import torch
from torch import Tensor
import numpy as np
def torchVectorizeChoice(pTensor: Tensor, device: torch.device, dim=1) -> Tensor:
    pTensorNum = pTensor.shape[1-dim]
    r = torch.rand((pTensorNum, 1), device=device)  # (p_array_num, 1)
    return torch.argmax((pTensor.cumsum(dim=dim) > r).float(), dim=dim)  # argmax will return the first element larger than r

def computeKS(F1, F2):
    return torch.max(torch.abs(F1 - F2))

def computeTV(prob1, prob2):
    return torch.sum(torch.abs(prob1 - prob2)).cpu().item() * 0.5

def computeCDF(probsOfAtoms):
    cdfsOfAtoms = torch.zeros_like(probsOfAtoms)
    cdfsOfAtoms[:, 0] = probsOfAtoms[:, 0]
    for i in range(1, probsOfAtoms.size(dim = 1)):
        cdfsOfAtoms[:, i] = cdfsOfAtoms[:, i - 1] + probsOfAtoms[:, i]
    return cdfsOfAtoms.cpu()

def randomPolicy(sizeOfStateSpace, sizeOfActionSpace, device):
    policy = np.random.uniform(0.1, 0.9, size=(sizeOfStateSpace, sizeOfActionSpace))
    policy = policy / (np.sum(policy, axis=1)[:, np.newaxis])
    policy = torch.as_tensor(policy, dtype=torch.float32, device=device)
    return policy