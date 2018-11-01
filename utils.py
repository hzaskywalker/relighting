import torch
from torch import nn
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def make_input(x, cuda=True, requires_grad=False):
    ## make x be float32
    if x is None: return x
    d_ = device if cuda else torch.device("cpu")
    x = torch.tensor(np.array(x, dtype='float32')).to(d_)
    if requires_grad:
        x.requires_grad_()
    return x

def make_output(x):
    if x.is_cuda:
        x = x.cpu()
    if x.requires_grad:
        x = x.detach()
    return x.numpy()

MI = make_input
MO = make_output

def save(path, dict, epochs):
    print('Saving.. ', path)
    #state = net.state_dict()
    if not os.path.isdir(path):
        os.mkdir(path)
    for i in dict:
        dict[i] = dict[i].state_dict()
    dict['epoch'] = epochs
    torch.save(dict, os.path.join(path, 'ckpt.t7'))

def resume(path, dict):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'ckpt.t7'))
    for i in dict:
        dict[i].load_state_dict(checkpoint[i])
    print('==> Loaded epoch: {}'.format(checkpoint['epoch']))

def resume_if_exists(path, net):
    if path is not None and os.path.exists(os.path.join(path, 'ckpt.t7')):
        resume(path, net)


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1)
    return kld

def reparametrize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)

from tensorboardX import SummaryWriter

class TensorboardHelper:
    def __init__(self, filename, mode='train'):
        self.filename = filename
        self.writer = SummaryWriter(filename)
        self.init(0, mode)

    def init(self, step, mode):
        self.step = step
        self.mode = mode

    def name(self, name):
        if self.mode is not None:
            return '{}_{}'.format(self.mode, name)
        else:
            return name

    def add_scalar(self, tag, value):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, self.step)

    def add_scalars(self, tag, values):
        """Log a scalar variable."""
        self.writer.add_scalars(tag, values, self.step)

    def add_image(self, tag, images):
        """Log a list of images."""

        img_summaries = []

        if images.shape[1] <= 3:
            images = images.transpose(0, 2, 3, 1)
        for i, img in enumerate(images):
            if img.shape[2] == 1:
                import cv2
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self.writer.add_image(self.name('%s/%d'%(tag, i)), img[None,:].transpose(0, 3, 1, 2), self.step)

    def tick(self):
        self.step += 1

class Visualizer(object):
    def __init__(self, exp_path, trainer=None, num_iters=None):
        self.path = exp_path
        self.tb = TensorboardHelper(exp_path)
        if trainer is not None:
            assert num_iters is not None, 'please provide the number of batches per epoch'
            self.tb.step = trainer.epoch * 600
            print(trainer.epoch)
            print(self.tb.step)
            print('starting from iteration', self.tb.step)

    def dfs(self, prefix, outputs):
        if outputs is None:
            return
        if type(outputs) is dict:
            for i in outputs:
                self.dfs('{}/{}'.format(prefix, i), outputs[i])
        else:
            outputs = np.array(outputs)
            if len(outputs.shape) == 4:
                self.tb.add_image(prefix, outputs)
            else:
                self.tb.add_scalar(prefix, outputs)

    def __call__(self, outputs):
        self.dfs('', outputs)
        self.tb.tick()

def viewDirection(weights, degree, size):
    ## dirs (1053 * 3)
    from data.coefs import coefs, angles
    degree = degree / 180. * np.pi

    mask = (angles <= degree)
    loc = coefs[mask]

    x = np.int32(np.round(loc[:,0] * size))
    y = np.int32(np.round(loc[:,1] * size))
    img = np.zeros((size, size))
    for x, y, w in zip(x, y, weights):
        img[y, x] = w
    img = img/img.max()
    return img