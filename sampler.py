## different kinds of sampler
import torch
from torch import nn

class Sampler(nn.Module):
    ## sample according to mask
    def __init__(self, sample_methods='sum'):
        nn.Module.__init__(self)
        self.sample_methods = sample_methods

    def forward(self, inps, mask, alpha):
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


class EncodeNet(nn.Module):
    def __init__(self, step, inp_dim=5, start=32):
        nn.Module.__init__(self)
        layers = []
        for i in range(step):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(inp_dim, start, 3, stride=2, padding=1),
                    nn.BatchNorm2d(start),
                    nn.ReLU(),
                )
            )
            inp_dim = start
            start = start * 2
        self.main = nn.Sequential(*layers)
        self.output_dim = inp_dim

    def forward(self, x):
        x = self.main(x)
        x = x.mean(dim=3).mean(dim=2)
        return x


def sample(inps, mask):
    ## inps (b, 1053, xxx)
    ## mask (1053, k)
    inps = inps.transpose(0, 1)
    size = inps.size()
    inps = inps.reshape(inps.size(0), -1)
    ans = mask.transpose(1, 0) @ inps
    ans = ans.reshape(-1, *size[1:]).transpose(0, 1).contiguous()
    ans = ans.view(ans.size(0), -1, *ans.size()[3:])
    return ans

class FixedDirectionNet(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)
        self.num_lights = num_lights
        self.K = num_samples
        self.mask = nn.Parameter(torch.ones(num_lights, num_samples))
        self.output_dim = num_samples * 5

    def forward(self, inps, identity, alpha=8):
        #inps (scene, num_lights, k, size, size)

        mask = torch.nn.functional.softmax(self.mask[:,identity:identity+1] * alpha, dim=0)
        #ans = (inps[:,:,None] * mask[None, :, :, None, None,None]).sum(dim=1)
        return sample(inps, mask)



class AdaptiveSampler2(nn.Module):
    ## with intermediate state
    ## example extract net: Unet
    ## example sample_directions: set of sampleNet/random net
    def __init__(self, extractor, next_direction, sample_method, number_samples=2):
        nn.Module.__init__(self)
        self.extractNet = extractor
        self.next_direction = next_direction
        self.sample_method = sample_method
        self.number_samples = number_samples

    def forward(self, inps, alpha=1.):
        state = (torch.rand(inps.size(0), self.extractNet.output_di, inps.size()[-2], inps.size()[-1]) * 0).to(inps.device)

        for i in range(self.number_samples):
            directions = self.next_direction(state, i)
            sampled_image = self.sample_method(inps, directions, alpha)

            new_state = self.extractNet(sampled_image)

            if i == 0:
                state = new_state
            else:
                state = torch.max(state, new_state)
        return state

def makeFixed(Light, num_samples):
    from networks2 import ImageFeatureExtractor
    extractor = ImageFeatureExtractor(4, 3, 32, 32)
    next_directon = FixedDirectionNet(len(Light), num_samples)
    sample_methods = Sampler('sum')
    return AdaptiveSampler2(extractor, next_direction, sample_methods, num_samples)