## code for the dynamic numbers of training data
## the goal today is just to finish the code part, as I don't have the chance and the power to run and debug the code..
## I think I have finished this part of work
## at least for now

"""
Multiple samples
To achieve the effect that with more samples, get better results
Relighter should be separated from the sampler

Intermediate result is part of the sampler but not he  
This is just for testing the multi output
"""
import torch
import numpy as np
from torch import nn

class UNet(nn.Module):
    def __init__(self, d, inp_dim, oup_dim):
        nn.Module.__init__(self)
        self.d = d
        self.inp_dim = inp_dim
        self.oup_dim = oup_dim
        if d > 0:
            self.down = nn.Sequential(
                nn.Conv2d(inp_dim, oup_dim, 3, stride=2,padding=1),
                nn.BatchNorm2d(oup_dim),
                nn.ReLU(),
            )

            self.relightA = UNet(d-1, oup_dim, oup_dim * 2)


            dim = self.relightA.oup_dim
            self.up = nn.Sequential(
                nn.ConvTranspose2d(dim, oup_dim, 3, stride=2,padding=1, output_padding=1),
                nn.BatchNorm2d(oup_dim),
                nn.ReLU()
            )
            self.oup_dim = oup_dim + inp_dim
        else:
            self.main = nn.Sequential(
                nn.Conv2d(inp_dim, inp_dim, 3, stride=1,padding=1),
                nn.BatchNorm2d(inp_dim),
                nn.ReLU(),
            )
            self.oup_dim = inp_dim

    def forward(self, x):
        if self.d == 0: 
            out = self.main(x)
        else:
            down = self.down(x)
            tmp = self.relightA(down)
            up = self.up(tmp)
            out = torch.cat((x, up), dim=1)
        return out

class NovelAngle(nn.Module):
    def __init__(self, oup_dim=128):
        nn.Module.__init__(self)

        self.main = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, oup_dim),
        )
        self.oup_dim = oup_dim

    def forward(self, x):
        return self.main(x)

class RelightUnit(nn.Module):
    def __init__(self, d, inp_dim, oup_dim):
        nn.Module.__init__(self)
        self.d = d
        if d > 0:
            self.down = nn.Sequential(
                nn.Conv2d(inp_dim, oup_dim, 3, stride=2,padding=1),
                nn.BatchNorm2d(oup_dim),
                nn.ReLU(),
            )

            self.relightA = RelightUnit(d-1, oup_dim, oup_dim * 2)


            dim = self.relightA.oup_dim
            self.up = nn.Sequential(
                nn.ConvTranspose2d(dim, oup_dim, 3, stride=2,padding=1, output_padding=1),
                nn.BatchNorm2d(oup_dim),
                nn.ReLU()
            )
            self.oup_dim = oup_dim + inp_dim
        else:
            self.relightA = NovelAngle()
            self.merge = nn.Sequential(
                nn.Conv2d(self.relightA.oup_dim + inp_dim, inp_dim, 3, stride=1,padding=1),
                nn.BatchNorm2d(inp_dim),
                nn.ReLU(),
            )
            self.oup_dim = inp_dim + inp_dim 

    def forward(self, x, w):
        if self.d == 0: 
            w = self.relightA(w)
            w = w[..., None,None].expand((w.size(0), w.size(1), x.size(2), x.size(3)))
            out = self.merge(torch.cat((x, w),dim=1))
            out = torch.cat((x, out), dim=1)
        else:
            down = self.down(x)
            tmp = self.relightA(down, w)
            up = self.up(tmp)
            out = torch.cat((x, up), dim=1)
        return out
    
class RelightNet(nn.Module):
    """
    relight the image with the input dimension oup_dim
    """
    def __init__(self, d, inp_dim, oup_dim):
        nn.Module.__init__(self)
        self.main = RelightUnit(d, inp_dim, oup_dim)
        self.post = nn.Sequential(
            nn.Conv2d(self.main.oup_dim, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 1, stride=1)
        )
    def forward(self, x, ws):
        return self.post(self.main(x, ws))

def expand_ws(inps, ws):
    inps = inps[:,None].expand((inps.size(0), ws.size(1),)+inps.size()[1:])
    inps = inps.contiguous().view(inps.size(0)*inps.size(1), *inps.size()[2:]) # b*num_relight, K, f, w, h
    return inps

class UnetExtractor(nn.Module):
    def __init__(self, d, inp_dim, output_dim):
        nn.Module.__init__(self)
        self.main = UNet(4, inp_dim, output_dim)
        self.output_dim = self.main.oup_dim

    def forward(self, inps):
        return self.main(inps)

    def init_state(self, inps):
        state = (torch.rand(inps.size(0), self.output_dim, inps.size()[-2], inps.size()[-1]) * 0).to(inps.device)
        return state

    def add_state(self, old_state, new_state):
        return torch.stack((old_state, new_state), dim=0).max(dim=0)[0]

class SetExtractor(UnetExtractor):
    def init_state(self, inps):
        state = UnetExtractor.init_state(self, inps)
        return state[:,None,:]
    
    def forward(self, inps):
        state = self.main(inps)
        return state

    def add_state(self, old_state, new_state):
        return torch.cat((old_state, new_state[:,None]), dim=1)

class SetRelighter(nn.Module):
    def __init__(self, d, inp_dim, oup_dim):
        nn.Module.__init__(self)
        self.main = RelightUnit(d, inp_dim, oup_dim)
        self.w_dim = 5
        self.novelAngle = NovelAngle(oup_dim=5)
        self.post = nn.Sequential(
            nn.Conv2d(self.main.oup_dim, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 1, stride=1)
        )
    def forward(self, x, ws):
        x = x[:,1:]
        w_dim = self.novelAngle(ws)
        dist = ((x[:, :, :self.w_dim] - w_dim[:,None,:, None, None])**2)
        weights = nn.functional.softmax(dist.sum(dim=2), dim=1)
        x = (x * weights[:,:,None,:,:]).sum(dim=1)
        return self.post(self.main(x, ws))

class RandomNextDirection(nn.Module):
    # the output is the mask

    def __init__(self, num_light):
        nn.Module.__init__(self)
        self.num_light = num_light 

    def forward(self, inps, times):
        ## ignore times
        idx = []
        out = torch.zeros(inps.size(0), self.num_light).to(inps.device)

        for i in range(len(inps)):
            idx.append( (np.random.randint(self.num_light),) )
        return idx

    def add_padding(self, inps, Light):
        to_pad = Light[None,:,:,None,None]
        to_pad = to_pad.expand(inps.size(0), to_pad.size(1), to_pad.size(2), inps.size(3), inps.size(4))
        inps = torch.cat((inps, to_pad), dim=2)
        return inps

    def sample(self, inps, mask):
        assert inps.shape[2] == 5
        ans = []
        for a, b in zip(inps, mask):
            ans.append( a[b[0]] )
        return torch.stack(ans, dim=0)

class FixedDirection(RandomNextDirection):
    def __init__(self, lights, dirs):
        nn.Module.__init__(self)
        dirs = dirs/180 * np.pi
        t = np.sin(dirs[:,0])
        x = np.cos(dirs[:,1]) * t
        y = np.sin(dirs[:,1]) * t
        dirs = np.stack((x, y), axis=1)
        dist = ((lights[None,:, :].cpu().detach().numpy() - dirs[:,None,:])**2).sum(axis=2)
        self.idx = dist.argmin(axis=1)
        #print(t[2])
        #print(lights[list(self.idx)], lights.min(dim=0), lights.max(dim=0))

    def forward(self, inps, times):
        assert times < len(self.idx)
        ans = [(self.idx[times],) for i in range(len(inps))]
        return ans

class FixedMaskDirector(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)
        self.mask = nn.Parameter(torch.rand(num_samples, num_lights))
        self.alpha = nn.Parameter(torch.tensor(1.))

    def forward(self, inps, times):
        mask = nn.functional.softmax(self.mask * self.alpha.detach(), dim=1)[times]
        return mask

    def sample(self, inps, mask):
        return (inps * mask[None,:,None,None,None]).sum(dim=1)

class EncodeNet(nn.Module):
    def __init__(self, step, inp_dim=5, start=32):
        nn.Module.__init__(self)
        layers = []
        for i in range(step):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(inp_dim, start, 3, stride=2, padding=1),
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

class AdatpiveMaskDecoder(nn.Module):
    def __init__(self, num_lights, num_samples, inp_dim=37):
        nn.Module.__init__(self)
        self.mask = nn.Parameter(torch.rand(num_samples, num_lights))
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.unet = []
        for i in range(4):
            net = EncodeNet(4, inp_dim=inp_dim, start=64)
            self.unet.append(
                nn.Sequential(
                    net,
                    nn.Linear(net.output_dim, num_lights)
                )
            )
        self.unet = nn.ModuleList(self.unet)

    def forward(self, inps, times):
        mask = self.unet[times](inps)
        mask = nn.functional.softmax(mask * self.alpha.detach(), dim=1)
        return mask

    def sample(self, inps, mask):
        return (inps * mask[:,:,None,None,None]).sum(dim=1)


def test_fixed_direction():
    from utils import getLight
    lights = getLight(45)
    dirs = [
        [21, 86],
        [32, 171],
        [31, 351],
        [24, 266],
    ]
    net = FixedDirection(lights, np.array(dirs))

def test_random_next_direction():
    a = RandomNextDirection(8)
    t = torch.zeros(100, 1).cuda()
    print(a(t))
#test_random_next_direction()
test_fixed_direction()