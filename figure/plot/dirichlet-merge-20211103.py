from math import gamma
import numpy.random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib
import os
print('Set the three positive values, a1, a2, a3...')
gamma_ind = 947
gamma_str = chr(947)
print(gamma_str,gamma_ind)
plt.switch_backend('agg')
plt.figure(figsize = (12,3.6))


p1 = np.array([0.0, (3.0**0.5) - 1.0])
p2 = np.array([-1.0, -1.0])
p3 = np.array([1.0, -1.0])


gama_values = [ [0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0] ]

i=1
for gama_value in gama_values:
    print("i=",i)
    print("gama_value:",gama_value)
    plt.subplot(1,3,i)
    plt.subplots_adjust(wspace=0.1)

    plt.plot([-1.0, 1.0, 0.0, -1.0], [-1.0, -1.0, (3.0**0.5) - 1.0, -1.0])
    # plt.plot([-1.0, 1.0, 0.0], [-1.0, -1.0, (3.0**0.5) - 1.0])
    raw_data = numpy.random.dirichlet([gama_value[0], gama_value[1], gama_value[2]], 1000)

    points = []
    for t1, t2, t3 in raw_data:
        points.append(p1 * t1 + p2 * t2 + p3 * t3) 
    points = np.array(points)
    print("points.shape:",points.shape)


    plt.title(f'Dirichlet({gama_value[0]:.1f},{gama_value[1]:.1f},{gama_value[2]:.1f}) distribution')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1)
    plt.scatter(points.T[0], points.T[1], edgecolors = '#ffffff', alpha = 0.5)
    plt.text(p1[0]-0.1, p1[1]+0.05, 'P1', fontsize = 16)
    plt.text(p2[0]-0.15, p2[1]-0.175, 'P2', fontsize = 16)
    plt.text(p3[0]-0.05, p3[1]-0.175, 'P3', fontsize = 16)
    plt.xticks([])
    plt.yticks([])
    i = i +1

plt.show()
savepath = f'result/dirichlet'
savename = f'dirichlet-distribution-mixed-data'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()
