## train the networks that can render arbitary length of the image, with arbitary kinds of samples
## we only train the relight net
import argparse
import tqdm
import numpy as np
import torch
from torch import nn
from utils import getLight, resume_if_exists, handle_data, Visualizer, save
from dataloader import get_train_data

from network2 import UnetExtractor, RandomNextDirection, RelightNet, expand_ws
from sampler import sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lights', type=int, default=1053)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--degree', type=int, default=45, choices=[45, 90])
    parser.add_argument('--sampler', type=str, default='original', choices=['original', 'gaussian', 'categorical', 'basicAdaptive'])
    parser.add_argument('--methods', type=str, default='sum', choices=['sum', 'sample'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_scene', type=int, default=4)
    parser.add_argument('--num_gt', type=int, default=18)
    parser.add_argument('--path', type=str, default='exp/max_relighter')
    parser.add_argument('--num_epochs', type=int, default=16)
    parser.add_argument('--dataset_path', type=str, default='/home/hza/data/rendering/cropped_64.hdf5')
    args = parser.parse_args()

    light = getLight(args.degree)
    args.num_light = len(light)
    data = get_train_data(num_workers = args.num_workers, batch_size=args.num_scene, path=args.dataset_path, degree=args.degree, num_split=args.num_gt)


    extractor = UnetExtractor(4, 5, 32)
    director = RandomNextDirection(args.num_light)
    relighter = RelightNet(4, extractor.output_dim, 64)

    module_list = nn.ModuleList([extractor, director, relighter])
    module_list.cuda()
    optim = torch.optim.Adam(module_list.parameters(), 0.0001) ## changed the learning rate

    def zipper():
        ## usually, visualizer should also be saved 
        return {'optim':optim,
            'extractor': extractor,
            'director': director,
            'relighter': relighter,
        }

    dic = zipper()
    resume_if_exists(args.path, dic)
    if 'epoch' in dic:
        start_epoch = dic['epoch']
    else:
        start_epoch = -1

    viewer = Visualizer(args.path)

    for i in range(start_epoch+1, args.num_epochs):
        b = data.__iter__()
        num_batchs = len(b)

        for batch_id in tqdm.trange(num_batchs):
            tmp = next(b)
            inps, gts, ws = handle_data(tmp, light)
            inps = director.add_padding(inps, light)

            """
            key part
            """
            num_samples = np.random.randint(2, 11)

            optim.zero_grad()
            state = extractor.init_state(inps)

            for j in range(num_samples):
                next_directions_mask = director(state, j)
                sampled_image = director.sample(inps, next_directions_mask)

                new_state = extractor(sampled_image)
                state = extractor.add_state(state, new_state)

            state = expand_ws(state, ws)
            ws = ws.view(-1, 2)
            assert state.shape[0] == ws.shape[0]
            output = relighter(state, ws)


            gts = gts.view(-1, *gts.size()[2:])
            assert output.shape == gts.shape 
            loss = ((output - gts)**2).mean()
            loss.backward()
            optim.step()

            if batch_id % 100 == 0:
                """
                visualization code
                """
                toshow = torch.cat((gts, output), dim=3).detach().cpu().detach().numpy()
                toshow = np.minimum(toshow, 1)
                toshow = np.maximum(toshow, -1)
                toshow = (toshow * 127.5 + 127.5).astype(np.uint8)
                viewer({'img': toshow, 'loss': loss.mean().cpu().detach().numpy()})

        save(args.path, zipper(), i)


if __name__ == '__main__':
    main()