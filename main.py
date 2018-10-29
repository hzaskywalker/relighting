import torch
from torch import nn
import tqdm
import argparse
import numpy as np
from networks import RelightNet, SampleNet
from data.coefs import dirs
from dataloader import get_train_data
from utils import resume_if_exists, save, Visualizer

Light = torch.tensor(np.float32(dirs)).cuda() ## 1053 * 3 * 8 * 8

class RelightNetwork(nn.Module):
    def __init__(self, num_lights, num_samples):
        nn.Module.__init__(self)
        self.sample = SampleNet(num_lights, num_samples)
        self.relighter = RelightNet(4, 5 * num_samples, 64)

    def forward(self, inps, ws, alpha=5):
        ## padd the light_direction 
        ##TODO: please make sure the input is normalized
        to_pad = Light[None,:,:,None,None]
        to_pad = to_pad.expand(inps.size(0), to_pad.size(1), to_pad.size(2), inps.size(3), inps.size(4))

        inps = torch.cat((inps, to_pad), dim=2) #(b, K', f, w, h)

        inps = self.sample(inps, alpha) #(b, K, f, w, h)

        inps = inps[:,None].expand((inps.size(0), ws.size(1),)+inps.size()[1:])
        inps = inps.contiguous().view(inps.size(0)*inps.size(1), *inps.size()[2:]) # b*num_relight, K, f, w, h
        ws = ws.view(-1, 2)

        return self.relighter(inps, ws)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lights', type=int, default=1053)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_scene', type=int, default=4)
    parser.add_argument('--num_gt', type=int, default=18)
    parser.add_argument('--path', type=str, default='exp')
    parser.add_argument('--num_epochs', type=int, default=16)
    parser.add_argument('--dataset_path', type=str, default='/home/hza/data/rendering/cropped.hdf5')
    args = parser.parse_args()

    net = RelightNetwork(args.num_lights, args.num_samples).cuda()
    optim = torch.optim.Adam(net.parameters(), args.lr)
    def zipper(net, optim):
        return {'optim':optim, 'net': net}

    resume_if_exists(args.path, zipper(net, optim))
    ## todo: tensorboard for relightling effects   
    data = get_train_data(num_workers = args.num_workers, batch_size=args.num_scene, path=args.dataset_path, num_light=args.num_lights, num_split=args.num_gt)
    viewer = Visualizer(args.path)

    for epochs in range(args.num_epochs):
        b = data.__iter__()

        num_batchs = len(b)
        for j in tqdm.trange(num_batchs):
            ids = (epochs + float(j)/num_batchs)
            T = 1 + 5 * (ids+ 0.0) ** 2.0

            tmp = next(b)
            inps = tmp[0].cuda()
            inps = (inps.permute(0, 1, 4, 2, 3).float() - 127.5)/127.5
            ws = []
            gts = []

            light_ids = tmp[1].numpy()
            for scene_id in range(len(inps)):
                w = []
                gt = []
                for k in range(args.num_gt):
                    #sampled = np.random.randint(args.num_lights)
                    sampled = light_ids[scene_id][k]
                    w.append(Light[sampled])
                    gt.append(inps[scene_id][sampled])
                ws.append(torch.stack(w, dim=0))
                gts.append(torch.stack(gt, dim=0))

            gts = torch.stack(gts, dim=0)
            ws = torch.stack(ws, dim=0)

            optim.zero_grad()
            out = net(inps, ws, alpha=T)

            gts = gts.view(gts.size(0)*gts.size(1), *gts.size()[2:])
            loss = nn.functional.mse_loss(out, gts)

            loss.backward()
            optim.step()

            if j % 100 == 0:
                toshow = torch.cat((gts, out), dim=3).detach().cpu().detach().numpy()
                toshow = np.minimum(toshow, 1)
                toshow = np.maximum(toshow, -1)
                toshow = (toshow * 127.5 + 127.5).astype(np.uint8)

                mask = nn.functional.softmax(net.sample.mask*T, dim=0).cpu().detach().numpy()
                inds = mask.argmax(axis=0)
                values = [mask[inds[i], i] for i in range(inds.shape[0])]
                if j == 0:
                    print(inds, values)
                viewer({'img': toshow})

        dict = zipper(net, optim)
        save(args.path, dict, epochs)

if __name__ == '__main__':
    main()