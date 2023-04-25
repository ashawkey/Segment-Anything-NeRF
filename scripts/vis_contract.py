import matplotlib.pyplot as plt
import torch
import numpy as np

def contract(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True) # [N, 1], [N, 1]
    scale = 1 / mag.repeat(1, C)
    scale.scatter_(1, idx, (2 - 1 / mag) / mag)
    cx = torch.where(mag < 1, x, x * scale)
    return cx.view(*shape, C)

def uncontract(x):
    # x: [..., C]
    shape, C = x.shape[:-1], x.shape[-1]
    x = x.view(-1, C)
    mag, idx = x.abs().max(1, keepdim=True) # [N, 1], [N, 1]
    scale = 1 / (2 - mag.repeat(1, C)).clamp(min=1e-8)
    scale.scatter_(1, idx, 1 / (2 * mag - mag * mag).clamp(min=1e-8))
    cx = torch.where(mag < 1, x, x * scale)
    return cx.view(*shape, C)


# # plot a line and a contracted line
x1 = torch.linspace(-4, 4, 100)
y1 = 1.1 * x1
x2 = torch.linspace(-4, 4, 100)
y2 = -0.5 * x1 - 0.1

points = torch.stack([torch.concat([x1, x2]), torch.concat([y1, y2])], dim=1)

# points = torch.rand(256, 2) * 10 - 5

cpoints = contract(points)

rpoints = uncontract(cpoints)

plt.plot(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), 'o', label='original')
plt.plot(cpoints[:, 0].cpu().numpy(), cpoints[:, 1].cpu().numpy(), 'o', label='contracted')
plt.plot(rpoints[:, 0].cpu().numpy(), rpoints[:, 1].cpu().numpy(), 'x', label='uncontracted')

plt.grid()
plt.legend()
plt.show()

