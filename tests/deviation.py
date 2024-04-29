import pygame
import torch
import numpy as np
from scipy.stats import norm
from torch import nn

pygame.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
).to(device)
# load model parameters
gen.load_state_dict(torch.load('../gen.pth'))

# load mean and standard deviation
with open('../distribution.txt') as f:
    mean = list(map(float, f.readline().split()))
    std = list(map(float, f.readline().split()))


def generate_input(n=1, dev=2.5):
    """
    :param n: number of generated inputs
    :return: random noise based on normal distribution
    """
    return np.array([norm.rvs(mean[i], std[i] * dev, n) for i in range(len(mean))]).T.reshape((w * h, len(mean), 1, 1))


# initialize display
w, h = 16, 8
display = pygame.display.set_mode((w * 64, h * 64 + 200))
dev = 0
font = pygame.font.Font(size=100)
inp = generate_input(w * h, dev)
inp = torch.tensor(inp, dtype=torch.float32).to(device)
imgs = (gen(inp).to('cpu').detach().numpy().transpose(0, 2, 3, 1) * 255).reshape((w, h, 64, 64, 3))
imgs = [[pygame.surfarray.make_surface(j) for j in i] for i in imgs]

while True:
    # handle events
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            exit()
        if ev.type == pygame.MOUSEBUTTONDOWN:
            dev = max(0, min(15, dev + {5: -0.25, 4: 0.25}[ev.button]))
            inp = generate_input(w * h, dev)
            inp = torch.tensor(inp, dtype=torch.float32).to(device)
            imgs = (gen(inp).to('cpu').detach().numpy().transpose(0, 2, 3, 1) * 255).reshape((w, h, 64, 64, 3))
            imgs = [[pygame.surfarray.make_surface(j) for j in i] for i in imgs]
    # draw images
    display.fill((255, 255, 255))
    for x in range(w):
        for y in range(h):
            display.blit(imgs[x][y], (x * 64, y * 64))
    txt = font.render(str(dev), True, (0, 0, 0))
    display.blit(txt, ((64 * w - txt.get_width()) // 2, 64 * h + (200 - txt.get_height()) // 2))
    # update display
    pygame.display.flip()
