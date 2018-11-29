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


class RelightNetWithMax(nn.Module):
    def __init__(self, d, inp_dim, oup_dim, mid_dim=32):
        self.unet = UNet(4, inp_dim, mid_dim)
        self.relight = RelightNet(d, self.unet.oup_dim, oup_dim)

    def forward(self, inps, ws):
        ans = []
        for i in range(inps.size(0)):
            k = inps[i]
            out, _ = self.unet(k).max(dim=0)
            ans.append(out)
        ans = torch.stack(ans, dim=0)
        return self.relight(ans, ws)


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

class SampleNet(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)
        self.num_lights = num_lights
        self.K = num_samples
        self.mask = nn.Parameter(torch.ones(num_lights, num_samples))
        self.output_dim = num_samples * 5

    def forward(self, inps, alpha=8):
        #inps (scene, num_lights, k, size, size)

        mask = torch.nn.functional.softmax(self.mask * alpha, dim=0)
        #ans = (inps[:,:,None] * mask[None, :, :, None, None,None]).sum(dim=1)
        return sample(inps, mask)




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

class AdaptiveSampleNet(nn.Module):
    def __init__(self, hidden_dim, num_lights):
        nn.Module.__init__(self)
        from resnet import resnet18
        self.main = resnet18(inp_dim=hidden_dim, num_classes=num_lights)

    def forward(self, x):
        return self.main(x)

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

class Sampler(nn.Module):
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



class AdaptiveSampling(nn.Module):
    def __init__(self, Lights, dist, sampler, num_samples=5, inp_dim=5):
        nn.Module.__init__(self)

        self.Lights = Lights

        model_list = []
        hidden_dim = 32

        self.dist = dist
        self.sampler = sampler

        if dist == 'categorical':
            output_dim = self.Lights.shape[0]
        else:
            output_dim = self.Lights.shape[1]

        self.high = float(self.Lights[:, 0].max())

        for i in range(num_samples):
            nets = EncodeNet(4, inp_dim * (i+1), hidden_dim)
            model_list.append(
                nn.Sequential(
                    nets,
                    nn.Linear(nets.output_dim, output_dim)
                )
            )
        self.models = nn.ModuleList(model_list)
        self.num_lights = self.Lights.shape[0]
        self.init_mask = nn.Parameter(torch.zeros(output_dim,))
        self.output_dim = num_samples * inp_dim
        self.num_samples = num_samples

    def get_mask(self, out):
        if self.dist == 'categorical':
            return out
        else:
            return (( torch.tanh( out[:,None] ) * self.high - self.Lights[None,:])**2).sum(dim=2)

    def forward(self, inps, alpha=1.0):
        pre = None
        self.extra_output = []
        logp = []
        for i in range(self.num_samples):
            if i == 0:
                mask = self.init_mask[None,:].expand(inps.size(0), self.init_mask.size(0))
            else:
                mask = self.models[i-1](pre)

            mask = self.get_mask(mask)
            self.extra_output.append(torch.nn.functional.softmax(mask*alpha, dim=1))

            sampled, logp_ = self.sampler(inps, mask, alpha)
            logp.append(logp_)

            if pre is None:
                pre = sampled
            else:
                pre = torch.cat((pre, sampled), dim=1)

        self.extra_output = torch.stack(self.extra_output, dim=2).cpu().detach().numpy()
        return pre, torch.stack(logp, dim=1).sum(dim=1)