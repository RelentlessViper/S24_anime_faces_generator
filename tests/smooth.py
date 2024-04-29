import pygame
import torch
import numpy as np
from scipy.stats import norm
from torch import nn

# create model
gen = nn.Sequential(
    nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
    nn.Sigmoid(),
)
# load model parameters
gen.load_state_dict(torch.load('../gen.pth'))

# load mean and standard deviation
with open('../distribution.txt') as f:
    mean = list(map(float, f.readline().split()))
    std = list(map(float, f.readline().split()))


def generate_input(n=1):
    """
    :param n: number of generated inputs
    :return: random noise based on normal distribution
    """
    return np.array([norm.rvs(mean[i], std[i] * 2.5, n) for i in range(len(mean))]).T.reshape((w * h, len(mean), 1, 1))


# initialize display
w, h = 16, 8
display = pygame.display.set_mode((w * 64, h * 64))
now = generate_input(w * h)

while True:
    nxt = generate_input(w * h)
    # images between now and nxt latent vectors
    for inp in np.linspace(now, nxt, 15):
        # handle events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                exit()
        # draw images
        inp = torch.tensor(inp, dtype=torch.float32)
        imgs = gen(inp).detach().numpy().transpose(0, 2, 3, 1) * 255
        for x in range(w):
            for y in range(h):
                img = pygame.surfarray.make_surface(imgs[x * h + y])
                display.blit(img, (x * 64, y * 64))
        # update display
        pygame.display.flip()
    now = nxt
