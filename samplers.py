## different kinds of sampler
import torch
from torch import nn

class Sampler(nn.Module):
    ## sample according to mask
    def __init__(self, sample_methods='sum'):
        nn.Module.__init__(self)
        self.sample_methods = sample_methods

    def forward(self, inps, mask, alpha=1.):
        mask = torch.softmax(mask * alpha, dim=1)
        if self.sample_methods == 'sum':
            sampled = (inps * mask[:,:,None,None,None]).sum(dim=1)
            logp = mask[:, 0] * 0
        else:
            logp = []
            sampled = []
            for i in range(len(mask)):
                dist = torch.distributions.Categorical(mask[i])
                loc = dist.sample()
                logp.append(dist.log_prob(loc))
                sampled.append(inps[i, loc])
            sampled = torch.stack(sampled, dim=0)
            logp = torch.stack(logp, dim=0)
        return sampled, logp

def add_padding(inps, Light):
    to_pad = Light[None,:,:,None,None]
    to_pad = to_pad.expand(inps.size(0), to_pad.size(1), to_pad.size(2), inps.size(3), inps.size(4))
    inps = torch.cat((inps, to_pad), dim=2)
    return inps

class FixedDirector(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)
        self.mask = torch.tensor
        self.mask = nn.Parameter(torch.ones(num_samples, num_lights))

    def forward(self, inps, times):
        return self.mask[times].expand(len(inps), self.mask[times].shape[0])