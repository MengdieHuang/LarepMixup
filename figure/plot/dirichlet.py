from math import gamma
import numpy.random as nprn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib
import os
print('Set the three positive values, a1, a2, a3...')
a1 = float(input('a1 = '))
a2 = float(input('a2 = '))
a3 = float(input('a3 = '))


raw_data = nprn.dirichlet([a1, a2, a3], 5000)
data = pd.DataFrame({'theta_1': raw_data.T[0], 'theta_2': raw_data.T[1],
                        'theta_3': raw_data.T[2]})
sb.pairplot(data)
plt.show()
# maggie
savepath = f'result/Dirichlet'
os.makedirs(savepath, exist_ok=True)
savename = f'a = [{a1}, {a2}, {a3}]-(1)'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()


p1 = np.array([0.0, (3.0**0.5) - 1.0])
p2 = np.array([-1.0, -1.0])
p3 = np.array([1.0, -1.0])
plt.figure(figsize = (5, 5))
plt.plot([-1.0, 1.0, 0.0, -1.0],
         [-1.0, -1.0, (3.0**0.5) - 1.0, -1.0])


points = []
for t1, t2, t3 in raw_data:
    points.append(p1 * t1 + p2 * t2 + p3 * t3) 
points = np.array(points)
#r'$\gamma$'
gamma_ind = 947
gamma_str = chr(947)
print(gamma_str,gamma_ind)

# plt.title('Dirichlet({0}, {1}, {2})'.format(a1, a2, a3), fontsize = 16)
plt.title(f'Dirichlet({gamma_str}) distribution with {gamma_str}=({a1},{a2},{a3})')

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.scatter(points.T[0], points.T[1], edgecolors = '#ffffff', alpha = 0.5)
plt.text(p1[0]-0.1, p1[1]+0.1, 'P1', fontsize = 16)
plt.text(p2[0]-0.1, p2[1]-0.15, 'P2', fontsize = 16)
plt.text(p3[0], p3[1]-0.15, 'P3', fontsize = 16)
plt.xticks([])
plt.yticks([])
plt.show()

#maggie
savepath = f'result/Dirichlet'
savename = f'a = [{a1}, {a2}, {a3}]-(2)'
plt.savefig(f'{savepath}/{savename}.png')
plt.close()
