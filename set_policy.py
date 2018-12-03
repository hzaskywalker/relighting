## give relight network, extractor 
import torch
import tqdm
from torch import nn
import numpy as np
import argparse

from utils import getLight, resume_if_exists, Visualizer, handle_data, viewDirection, save
from dataloader import get_train_data
from network2 import UnetExtractor, RelightNet, expand_ws
from samplers import Sampler, FixedDirector, add_padding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='exp/sampler')
    parser.add_argument('--num_sample', type=int, default=4)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--degree', type=int, default=45)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_scene', type=int, default=4)
    parser.add_argument('--num_gt', type=int, default=18)
    parser.add_argument('--num_epochs', type=int, default=16)
    parser.add_argument('--dataset_path', type=str, default='/home/hza/data/rendering/cropped_64.hdf5')
    args = parser.parse_args()

    light = getLight(args.degree)

    sampler = Sampler(sample_methods='random')

    extractor = UnetExtractor(4, 5, 32).cuda()
    relighter = RelightNet(4, extractor.output_dim, 64).cuda()

    director = FixedDirector(len(light), args.num_sample).cuda()

    net_dic = torch.load('/home/hza/rendering/exp/max_relighter/ckpt.t7')

    extractor.load_state_dict(net_dic['extractor'])
    relighter.load_state_dict(net_dic['relighter'])
    extractor.eval()
    relighter.eval()

    optim = torch.optim.Adam(director.parameters(), 0.0001)

    def zipper():
        ## usually, visualizer should also be saved 
        return {
            'optim': optim,
            'director': director,
        }
    dic = zipper()
    resume_if_exists(args.path, dic)

    if 'epoch' in dic:
        start_epoch = dic['epoch']
    else:
        start_epoch = -1
    viewer = Visualizer(args.path)

    light = getLight(45)
    args.num_light = len(light)
    data = get_train_data(num_workers = args.num_workers, batch_size=args.num_scene, path=args.dataset_path, degree=args.degree, num_split=args.num_gt)

    for epoch_id in range(start_epoch+1, args.num_epochs):

        b = data.__iter__()
        num_batchs = len(b)

        for batch_id in tqdm.trange(num_batchs):
            tmp = next(b)
            inps, gts, ws = handle_data(tmp, light)
            inps = add_padding(inps, light)

            num_samples = args.num_sample

            optim.zero_grad()
            state = extractor.init_state(inps)

            logp_ = 0
            masks = []
            for j in range(num_samples):
                next_directions_mask = director(state, j)
                masks.append(next_directions_mask)
                sampled_image, logp = sampler(inps, next_directions_mask)
                logp_ = logp_ + logp

                new_state = extractor(sampled_image)
                state = extractor.add_state(state, new_state)

            state = expand_ws(state, ws)
            masks = torch.stack(masks, dim=1).detach()[0].transpose(1, 0)

            ws = ws.view(-1, 2)
            assert state.shape[0] == ws.shape[0]
            output = relighter(state, ws)


            gts = gts.view(-1, *gts.size()[2:])
            assert output.shape == gts.shape 
            loss = ((output - gts)**2).mean(dim=3).mean(dim=2).mean(dim=1)#nn.functional.mse_loss(out, gts)
            loss = loss.detach().view(len(inps), -1)

            adv = -loss.mean(dim=1)
            prob_loss = -(adv * logp_).mean()

            prob_loss.backward()
            optim.step()

            if batch_id % 100 == 0:
                """
                visualization code
                """
                toshow = torch.cat((gts, output), dim=3).detach().cpu().detach().numpy()
                toshow = np.minimum(toshow, 1)
                toshow = np.maximum(toshow, -1)
                toshow = (toshow * 127.5 + 127.5).astype(np.uint8)

                mask = [ nn.functional.softmax(masks, dim=0).cpu().detach().numpy() ]
                imgs = []
                for mask in mask:
                    #inds = mask.argmax(axis=0)
                    #values = [mask[inds[i], i] for i in range(inds.shape[0])]
                    tmp = []
                    for j in range(mask.shape[1]):
                        tmp.append( viewDirection(mask[:, j], args.degree, 31) )
                    tmp = np.concatenate(tmp, axis=1)
                    imgs.append(tmp)
                imgs = np.float32(np.concatenate(imgs))[None,None,:]

                viewer({'img': toshow, 'loss': loss.mean().cpu().detach().numpy(), 'prob_loss': prob_loss.mean().cpu().detach().numpy(), 'attention': imgs})

        save(args.path, zipper(), epoch_id)


if __name__ == '__main__':
    main()