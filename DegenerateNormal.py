import torch
class DegenerateNormal:
    def __init__(self, loc, cov, device):
        self.device = device
        self.loc = loc
        self.cov = cov
        self.dim = loc.size(dim=0)
        u, s, v = torch.svd(cov)
        self.u = u
        self.Gamma = torch.diag(torch.sqrt(s))
        self.trans = torch.matmul(self.u, self.Gamma)
    
    def sample(self, numOfSamples = 1):
        sample = torch.normal(torch.zeros(numOfSamples, self.dim, device = self.device))
        return torch.matmul(self.trans, sample.T).T + self.loc