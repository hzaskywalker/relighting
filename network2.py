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
    def __init__(self):
        nn.Module.__init__(self)

        self.main = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
        )
        self.oup_dim = 128

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

class ImageFeatureExtractor(nn.Module):
    def __init__(self, d, inp_dim, oup_dim, mid_dim=32):
        pass

    def forward(self, inps):
        return self.unet(inps)

class RelightNetwork(nn.Module):
    def __init__(self, num_lights, num_samples=10, sample_methods='sum', Light=None):
        nn.Module.__init__(self)
        maskNet = SampleNet(num_lights, num_samples)
        feature = ImageFeatureExtractor(4, self.sample.output_dim, 32)
        self.sampler = maskNet
        self.relighter = RelightNet(4, self.sample.output_dim, 64)
        self.Light = Light

    def expand_ws(self, inps, ws):
        inps = inps[:,None].expand((inps.size(0), ws.size(1),)+inps.size()[1:])
        inps = inps.contiguous().view(inps.size(0)*inps.size(1), *inps.size()[2:]) # b*num_relight, K, f, w, h
        return inps

    def forward(self, inps, ws, alpha=5, number=10):
        if self.train:
            number = np.random.random(1, 11)

        inps = self.sampler(inps, alpha) #(b, K, f, w, h)
        inps = self.add_padding(inps)
        inps = self.expand_ws(inps, ws)

        ws = ws.view(-1, 2)
        return self.relighter(inps, ws)



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
        #idx = torch.tensor(np.array(idx)).to(inps.device).long()
        #out.scatter_(1, idx, torch.ones_like(idx).to(idx.device).float())
        #return out.transpose(0, 1)
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


def test_random_next_direction():
    a = RandomNextDirection(8)
    t = torch.zeros(100, 1).cuda()
    print(a(t))
#test_random_next_direction()