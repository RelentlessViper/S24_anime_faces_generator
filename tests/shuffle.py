import torch
import pygame
import numpy as np
from scipy.stats import norm
from torch import nn

pygame.init()

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
    return np.array([norm.rvs(mean[i], std[i] * 3.5, n) for i in range(len(mean))]).T.reshape((w * h, len(mean), 1, 1))


# initialize display, latent vectors and generated images
w, h = 16, 8
images = pygame.Surface((w * 64, h * 64))
shuffle = pygame.Surface((w * 64, 200))
display = pygame.display.set_mode((w * 64, h * 64 + shuffle.get_height()))
vectors = generate_input(w * h)
generated = torch.tensor(vectors, dtype=torch.float32)
generated = (gen(generated).detach().numpy().transpose(0, 2, 3, 1) * 255).reshape((w, h, 64, 64, 3))
generated = [[pygame.surfarray.make_surface(j) for j in i] for i in generated]
vectors = vectors.reshape((w, h, len(mean), 1, 1))
[images.blit(generated[x][y], (x * 64, y * 64)) for x in range(w) for y in range(h)]
chosen = []

# + and = symbols in pygame.Surface format
font = pygame.font.Font(size=64)
plus = font.render('+', True, (0, 0, 0))
eq = font.render('=', True, (0, 0, 0))

while True:
    # handle events
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            exit()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            x, y = ev.pos
            x, y = x // 64, y // 64
            # add to chosen list or remove from it
            if x < w and y < h:
                if (x, y) in chosen:
                    chosen.remove((x, y))
                elif len(chosen) < w // 2 - 1:
                    chosen.append((x, y))
    # redraw display
    display.blit(images, (0, 0))
    sh = pygame.Surface((2 * len(chosen) * 64 + 64, 64))
    shuffle.fill((255, 255, 255))
    sh.fill((255, 255, 255))
    for i, (x, y) in enumerate(chosen):
        pygame.draw.rect(display, (255, 0, 0), (x * 64, y * 64, 64, 64), 3)
        sh.blit(generated[x][y], (2 * i * 64, 0))
        sym = plus if i < len(chosen) - 1 else eq
        sh.blit(sym, (2 * i * 64 + 64 + (64 - sym.get_width()) // 2, (64 - sym.get_height()) // 2))
    if chosen:
        res = np.array([vectors[x][y] for x, y in chosen])
        res = torch.tensor(res.sum(axis=0) / len(res), dtype=torch.float32)
        res = gen(res).detach().numpy().transpose(1, 2, 0) * 255
        res = pygame.surfarray.make_surface(res)
        sh.blit(res, (2 * len(chosen) * 64, 0))
        shuffle.blit(sh, ((shuffle.get_width() - sh.get_width()) // 2, (shuffle.get_height() - sh.get_height()) // 2))
    display.blit(shuffle, (0, 64 * h))
    # update display
    pygame.display.flip()
