from logging import error
from math import gamma
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib
import os
from scipy.stats import beta

print('Set the two positive values, a1, a2 ...')
a1 = float(input('a1 = '))
a2 = float(input('a2 = '))
beta_str = 946
beta_str = chr(946)
# print(beta_str)
plt.switch_backend('agg')

x = np.linspace(0, 1, 1002)[1:-1]
alpha_beta_value = [a1,a2]
print(alpha_beta_value)
dist = beta(alpha_beta_value[0], alpha_beta_value[1])
dist_y = dist.pdf(x)
plt.plot(x, dist_y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (alpha_beta_value[0], alpha_beta_value[1]))


plt.title(f'Beta({beta_str}) distribution with {beta_str}=({a1},{a2})')
plt.xlim(0, 1)
plt.ylim(0, 2.5)
plt.show()

savepath = f'result/beta'
savename = f'beta[{a1},{a2}]-distribution]'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


# draw beta mixup

p1 = np.array([-1.0, 0])
p2 = np.array([1.0, 0])
plt.figure(figsize = (5, 5))
plt.plot([-1.0, 1.0], [0, 0])

raw_data = numpy.random.beta(a1, a2, 50)
print("raw_data.len:",len(raw_data))
print("raw_data.shape:",raw_data.shape)
# print("raw_data:",raw_data)
# raise error

points = []
for t in raw_data:
    # print("t:",t)
    points.append( p1 * t + p2 * (1.0 - t) ) 
points = np.array(points)
print("points.shape:",points.shape)
# print("points:",points)


plt.title(f'Beta({beta_str}) distribution with {beta_str}=({a1},{a2})')

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)

plt.scatter(points.T[0], points.T[1], edgecolors = '#ffffff', alpha = 0.5)

plt.text(p1[0]-0.1, p1[1]-0.15, 'P1', fontsize = 16)
plt.text(p2[0]-0.1, p2[1]-0.15, 'P2', fontsize = 16)
plt.xticks([])
plt.yticks([])
plt.show()

savepath = f'result/beta'
savename = f'beta[{a1},{a2}]-mixed data]'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()
