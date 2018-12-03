"""
finished the test code
"""
import torch
from torch import nn
import tqdm

from dataloader import CustomTensorDataset
from torch.utils.data import Dataset, DataLoader

class TesterTensorDataset(CustomTensorDataset):
    def __init__(self, path, degree, num_split):
        CustomTensorDataset.__init__(self, path, degree, num_split, length=100)

class Tester:
    def __init__(self,  path, degree, num_split,
            batch_size  = 4,
            num_workers = 2
        ):

        self.num_split = num_split
        self.test_data = TesterTensorDataset(path, degree, num_split)
        self.loader = DataLoader(self.test_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)

    def __call__(self, func, Light, num=-1):
        b = self.loader.__iter__()
        num_batchs = len(b)
        if num != -1:
            num_batchs = min(num_batchs, num)

        s = 0
        for j in tqdm.trange(num_batchs):

            tmp = next(b)
            inps = tmp[0].cuda()
            inps = (inps.permute(0, 1, 4, 2, 3).float() - 127.5)/127.5
            ws = []
            gts = []

            light_ids = tmp[1].numpy()

            for scene_id in range(len(inps)):
                w = []
                gt = []
                for k in range(self.num_split):
                    sampled = light_ids[scene_id][k]
                    w.append(Light[sampled])
                    gt.append(inps[scene_id][sampled])
                ws.append(torch.stack(w, dim=0))
                gts.append(torch.stack(gt, dim=0))

            gts = torch.stack(gts, dim=0)
            ws = torch.stack(ws, dim=0)

            out = func(inps, ws)
            loss = ((out - gts)**2).mean(dim=3).mean(dim=2).mean(dim=1)
            s = s + loss.mean().detach().cpu().numpy()

        return s/num_batchs

def testNotFixLength(path='exp/max_relighter/ckpt.t7'):
    from network2 import UnetExtractor, RelightNet, RandomNextDirection, expand_ws
    from utils import getLight

    degree = 45

    light = getLight(degree)

    extractor = UnetExtractor(4, 5, 32).cuda()
    relighter = RelightNet(4, extractor.output_dim, 64).cuda()

    net_dic = torch.load(path)
    extractor.load_state_dict(net_dic['extractor'])
    relighter.load_state_dict(net_dic['relighter'])

    extractor.eval()
    relighter.eval()

    tester = Tester(path='/home/hza/data/rendering/cropped_test_64.hdf5', degree=degree, num_split=18, batch_size=4)

    director = RandomNextDirection(len(light))
    #from train_baseline import director

   
    def make_func(num_samples):
        def func(inps, ws):
            state = extractor.init_state(inps)
            inps = director.add_padding(inps, light)

            for j in range(num_samples):
                next_directions_mask = director(state, j)
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

    #print( tester(make_func(2), light, num=100) )
    #print( tester(make_func(5), light, num=100) )
    #print( tester(make_func(10), light, num=100) )
    print( tester(make_func(4), light, num=100) )

def testBaseline(path='exp/relight_baseline/ckpt.t7'):
    from network2 import UnetExtractor, RelightNet, RandomNextDirection, expand_ws
    from train_baseline import director
    from utils import getLight

    degree = 45

    light = getLight(degree)

    relighter = RelightNet(4, 20, 64).cuda()

    net_dic = torch.load(path)
    relighter.load_state_dict(net_dic['relighter'])

    relighter.eval()

    from train_baseline import director

    tester = Tester(path='/home/hza/data/rendering/cropped_test_64.hdf5', degree=degree, num_split=18, batch_size=4)

    def make_func(num_samples):
        def func(inps, ws):
            state = []
            inps = director.add_padding(inps, light)

            for j in range(num_samples):
                next_directions_mask = director(inps, j)
                sampled_image = director.sample(inps, next_directions_mask)

                state.append(sampled_image)
            state = torch.cat(state, dim=1)

            state = expand_ws(state, ws)
            ws = ws.view(-1, 2)
            assert state.shape[0] == ws.shape[0]
            output = relighter(state, ws)
            output = output.view(len(inps), -1, *output.size()[1:])
            return output
        return func

    print( tester(make_func(4), light, num=1000) )

def testSetSeT(path='exp/set_set/ckpt.t7'):
    from network2 import SetExtractor, SetRelighter, RandomNextDirection, expand_ws
    from utils import getLight

    degree = 45

    light = getLight(degree)

    extractor = SetExtractor(4, 5, 32).cuda()
    relighter = SetRelighter(4, extractor.output_dim, 64).cuda()

    net_dic = torch.load(path)
    extractor.load_state_dict(net_dic['extractor'])
    relighter.load_state_dict(net_dic['relighter'])

    extractor.eval()
    relighter.eval()

    tester = Tester(path='/home/hza/data/rendering/cropped_test_64.hdf5', degree=degree, num_split=18, batch_size=4)

    director = RandomNextDirection(len(light))
    #from train_baseline import director

   
    def make_func(num_samples):
        def func(inps, ws):
            state = extractor.init_state(inps)
            inps = director.add_padding(inps, light)

            for j in range(num_samples):
                next_directions_mask = director(state, j)
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

    #print( tester(make_func(2), light, num=100) )
    #print( tester(make_func(5), light, num=100) )
    #print( tester(make_func(10), light, num=100) )
    #print( tester(make_func(4), light, num=100) )

if __name__ == '__main__':
    #testNotFixLength()
    testBaseline()
    #testSetSeT()