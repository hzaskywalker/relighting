import torch
from torch import nn

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


class SampleNet(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)
        self.num_lights = num_lights
        self.K = num_samples
        self.mask = nn.Parameter(torch.randn(num_lights, num_samples))

    def forward(self, inps, alpha=8):
        #inps (scene, num_lights, k, size, size)

        mask = torch.nn.functional.softmax(self.mask * alpha, dim=0)
        #ans = (inps[:,:,None] * mask[None, :, :, None, None,None]).sum(dim=1)
        inps = inps.transpose(0, 1)
        size = inps.size()
        inps = inps.reshape(inps.size(0), -1)
        ans = mask.transpose(1, 0) @ inps
        ans = ans.reshape(-1, *size[1:]).transpose(0, 1).contiguous()
        ans = ans.view(ans.size(0), -1, *ans.size()[3:])
        return ans


class AdaptiveSampling(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)

    def forward(self, memory, alpah=8):
        pass