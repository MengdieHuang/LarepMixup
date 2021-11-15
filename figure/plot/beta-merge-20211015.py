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
beta_str = 946
beta_str = chr(946)
# print(beta_str)
plt.switch_backend('agg')
plt.figure(figsize = (4, 4))

x = np.linspace(0, 1, 1002)[1:-1]

# alpha_beta_values = [ [0.5,0.5], [1,1], [2,2] ]
alpha_beta_values = [ [1,1] ]

for alpha_beta_value in alpha_beta_values:
    print(alpha_beta_value)
    dist = beta(alpha_beta_value[0], alpha_beta_value[1])
    dist_y = dist.pdf(x)
    plt.plot(x, dist_y, c = 'green',label=r'$\beta=(%.1f,%.1f)$' % (alpha_beta_value[0], alpha_beta_value[1]))

plt.title(f'Beta({beta_str}) distribution')
# plt.title(f'Uniform(0,1) distribution')
plt.xlim(0, 1)
plt.ylim(0, 2.5)
# plt.ylim(0, 2)
plt.legend(loc = 'upper center')
# plt.legend(loc = 'lower center')
plt.show()

savepath = f'result/beta'
savename = f'beta-distribution-20211112'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()
# raise error


# draw beta mixup
plt.figure(figsize = (4, 4))

p1 = np.array([-1, 0])
p2 = np.array([1, 0])

i = 1
for alpha_beta_value in alpha_beta_values:
    print("i=",i)
    print("alpha_beta_value:",alpha_beta_value)
    plt.subplot(3,1,i)
    plt.subplots_adjust(hspace=0.5)
    plt.plot([-1, 1], [0, 0])
    raw_data = numpy.random.beta(alpha_beta_value[0], alpha_beta_value[1], 60)

    points = []
    for t in raw_data:
        points.append( p1 * t + p2 * (1.0 - t) ) 
    points = np.array(points)
    print("points.shape:",points.shape)
    plt.title(f'Beta({alpha_beta_value[0]:.1f},{alpha_beta_value[1]:.1f}) distribution')
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.2, 1.2)
    plt.scatter(points.T[0], points.T[1], edgecolors = '#ffffff', alpha = 0.5)
    plt.text(p1[0]-0.25, p1[1]-0.25, 'P1', fontsize = 12)
    plt.text(p2[0]+0.065, p2[1]-0.25, 'P2', fontsize = 12)
    plt.xticks([])
    plt.yticks([])

    i = i +1

plt.show()
savepath = f'result/beta'
savename = f'beta-distribution-mixed-data'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()
