## give relight network, extractor 
import torch
import tqdm
from torch import nn
import numpy as np
import argparse

from utils import getLight, resume_if_exists, Visualizer, handle_data, viewDirection, save
from dataloader import get_train_data
from network2 import UnetExtractor, RelightNet, expand_ws, RandomNextDirection
from samplers import Sampler, FixedDirector, add_padding

## directions, 

## start from V, as I don't have a clear about how to evaluate a
class V(nn.Module):
    def __init__(self, inp_dim, num_directions):
        nn.Module.__init__(self)
        from resnet import resnet18
        self.main = resnet18(inp_dim=inp_dim, num_classes=num_directions)

    def forward(self, state):
        return self.main(state)



def getLoss(state, ws, gts, relighter):
    batch_size = len(state)
    state = expand_ws(state, ws)

    ws = ws.view(-1, 2)
    assert state.shape[0] == ws.shape[0]
    output = relighter(state, ws)


    gts = gts.view(-1, *gts.size()[2:])
    assert output.shape == gts.shape 
    loss = ((output - gts)**2).mean(dim=3).mean(dim=2).mean(dim=1)#nn.functional.mse_loss(out, gts)
    loss = loss.detach().view(batch_size, -1)
    return loss


class SampleVDirector(RandomNextDirection):
    def __init__(self, value):
        nn.Module.__init__(self)
        self.value = value

    def forward(self, state, j):
        if j == 0:
            return [(np.random.randint(541),) for i in range(len(state))]
        ans = self.value(state)
        print(ans.max(dim=1))
        exit(0)
        return ans.argmax(dim=1)


def test(data, light, value, extractor, relighter):
    b = data.__iter__()
    num_batchs = len(b)

    director = SampleVDirector(value)
    director2 = RandomNextDirection(len(light))

    loss = 0

    from tester import Tester
    tester = Tester(path='/home/hza/data/rendering/cropped_test_64.hdf5', degree=45, num_split=18, batch_size=4)
    def make_func(director):
        def func(inps, ws):
            state = extractor.init_state(inps)
            inps = add_padding(inps, light)
            for j in range(4):
                next_directions_mask = director(state*0, j)
                print(next_directions_mask)
                sampled_image = director.sample(inps, next_directions_mask)

                new_state = extractor(sampled_image)
                state = extractor.add_state(state, new_state)

            state = expand_ws(state, ws)
            ws = ws.view(-1, 2)
            assert state.shape[0] == ws.shape[0]
            output = relighter(state, ws)
            output = output.view(len(inps), -1, *output.size()[1:])
            return output
        return func
    print(tester(make_func(director), light, num=100))
    print(tester(make_func(director), light, num=100))
    print(tester(make_func(director2), light, num=100))
    print(tester(make_func(director2), light, num=100))

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

    extractor = UnetExtractor(4, 5, 32).cuda()
    relighter = RelightNet(4, extractor.output_dim, 64).cuda()

    director = RandomNextDirection(len(light))

    net_dic = torch.load('/home/hza/rendering/exp/max_relighter/ckpt.t7')

    extractor.load_state_dict(net_dic['extractor'])
    relighter.load_state_dict(net_dic['relighter'])
    extractor.eval()
    relighter.eval()

    # value to train
    value = V(extractor.output_dim, len(light)).cuda()

    optim = torch.optim.Adam(value.parameters(), 0.001)

    def zipper():
        ## usually, visualizer should also be saved 
        return {
            'optim': optim,
            'value': value,
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

    test(data, light, value, extractor, relighter)
    exit(0)

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

            light_ids = tmp[1].numpy()
            loss = 0

            for j in range(num_samples):
                next_directions_mask = director(state, j)
                sampled_image = director.sample(inps, next_directions_mask)

                new_state = extractor(sampled_image)
                state = extractor.add_state(state, new_state)

                loss_j = getLoss(state, ws, gts, relighter).detach()
                predict_j = value(state)

                res = []
                for scene_id in range(len(light_ids)): 
                    for k in range(len(light_ids[0])):
                        sampled = light_ids[scene_id][k]
                        res.append( predict_j[scene_id][sampled] )
                res = torch.stack(res, dim=0)
                res = res.reshape(len(light_ids), len(light_ids[0]))
                assert res.shape == loss_j.shape

                loss = loss + ((res - loss_j)**2).mean()


            loss.backward()
            optim.step()

            if batch_id % 100 == 0:
                viewer({'loss': loss.mean().cpu().detach().numpy()})

        save(args.path, zipper(), epoch_id)


if __name__ == '__main__':
    main()