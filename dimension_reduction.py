# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 08:59:16 2026

@author: gerar
"""


import openpnm as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from joblib import Parallel, delayed

num_x = 5
num_y = 5
num_z = 5

spacing = 1e-4
pn = op.network.Demo(shape=[num_x, num_y, num_z], spacing=spacing)
air = op.phase.Air(network=pn)
N_X = 425

N_samples = 10000

X_raw = np.zeros((N_samples, N_X))
Y_K = np.zeros((N_samples, 1))
Y_mean = np.zeros((N_samples, 1))
Y_std = np.zeros((N_samples, 1))



    

def run_simulation(i):
    try:
        pn = op.network.Cubic(shape=[num_x, num_y, num_z], spacing=1e-4)
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.regenerate_models()
        air = op.phase.Air(network=pn)

        air['pore.contact_angle'] = 120
        air['pore.surface_tension'] = 0.072

        phase = op.phase.Phase(network=pn)
        phase['pore.viscosity'] = 1
        phase.add_model_collection(op.models.collections.physics.basic)

        phase['throat.contact_angle'] = 120
        phase['throat.surface_tension'] = 0.072
        phase['throat.diffusivity'] = 1.98e-5

        phase.regenerate_models()

        f = op.models.physics.capillary_pressure.washburn
        air.add_model(propname='throat.entry_pressure',
                      model=f,
                      surface_tension='throat.surface_tension',
                      contact_angle='throat.contact_angle',
                      diameter='throat.diameter')

        pn['pore.volume'] = 0.0

        drn = op.algorithms.Drainage(network=pn, phase=air)
        drn.set_inlet_BC(pores=pn.pores('left'))
        drn.run()

        ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
        ip.set_inlet_BC(pores=pn.pores('left'))
        ip.run()
        data_ip = ip.pc_curve()

        inlet = pn.pores('left')
        outlet = pn.pores('right')
        flow = op.algorithms.StokesFlow(network=pn, phase=phase)
        flow.set_value_BC(pores=inlet, values=1)
        flow.set_value_BC(pores=outlet, values=0)
        flow.run()
        phase.update(flow.soln)

        Q = flow.rate(pores=inlet, mode='group')[0]
        A = op.topotools.get_domain_area(pn, inlets=inlet, outlets=outlet)
        L = op.topotools.get_domain_length(pn, inlets=inlet, outlets=outlet)
        K = Q * L / A
        K_mD = K / 0.98e-12 * 1000

        x_row = data_ip.pc
        y_K = K_mD
        y_mean = np.mean(pn['throat.diameter'])
        y_std = np.std(pn['throat.diameter'])

        print(f'Sample {i} done — K = {K_mD:.3f} mD')
        return (x_row, y_K, y_mean, y_std)

    except Exception as e:
        print(f'Sample {i} failed: {e}')
        return None

results = Parallel(n_jobs=-1)(delayed(run_simulation)(i) for i in range(N_samples))

X_raw  = np.array([r[0] for r in results])
Y_K    = np.array([r[1] for r in results]).reshape(-1, 1)
Y_mean = np.array([r[2] for r in results]).reshape(-1, 1)
Y_std  = np.array([r[3] for r in results]).reshape(-1, 1)

Y_raw = np.hstack((Y_K, Y_mean, Y_std))


X = X_raw - np.mean(X_raw, axis=0)

Sigma = X.T @ X
lam, W = np.linalg.eigh(Sigma)
idx = lam.argsort()[::-1]
lam = lam[idx]
W = W[:, idx]



info = lam/np.sum(lam)
cum = np.cumsum(info)

k = 75

print(len(lam[cum < 0.9]))


T_X = X @ W[:,:k]

op.visualization.set_mpl_style()



fig, ax = plt.subplots(3, 1, dpi=300, figsize=(5, 6))
ax[0].hist(Y_K, color='pink', bins=20)
ax[0].set_xlabel('Absolute Permeability (mD)')
ax[1].hist(Y_mean*1e3, color='magenta', bins=20)
ax[1].set_xlabel('Throat Size Mean (mm)')
ax[2].hist(Y_std*1e3, color='cyan', bins=20)
ax[2].set_xlabel('Throat Size StDev (mm)')
plt.show()




fig, ax = plt.subplots(dpi=300, figsize=(6,4))
ax.plot(np.linspace(1, len(lam), len(lam)), info, label='Individual Information', color='blue')
ax.plot(np.linspace(1, len(lam), len(lam)), cum, label='Cumulative Information', color='purple')
ax.axhline(0.9, 0, len(lam), c='grey', linestyle='dotted')
ax.axhline(0.95, 0, len(lam), c='black', linestyle='dotted')
text = '90% info retained at PC = ' + str(len(lam[cum < 0.9]))
ax.text(100, 0.875, text, fontsize=10, c='grey', bbox=dict(
        facecolor='white',
        alpha=1,
        edgecolor='none',
        pad=1
    ))
text = '95% info retained at PC = ' + str(len(lam[cum < 0.95]))
ax.text(200, 0.935, text, fontsize=10, c='black', bbox=dict(
        facecolor='white',
        alpha=1,
        boxstyle='round',
        edgecolor='none',
        pad=0.25
    ))
ax.set_ylabel('% of information')
ax.set_xlabel('Number of PCs')
ax.legend()
plt.show()



fig, ax = plt.subplots(dpi=300)
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y_K, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Absolute Permeability (mD)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()

fig, ax = plt.subplots(dpi=300)
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y_mean, cmap='magma', alpha=0.5)
fig.colorbar(scatter, label='Average Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()

fig, ax = plt.subplots(dpi=300)
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y_std, cmap='viridis', alpha=0.5)
fig.colorbar(scatter, label='Standard Deviation of Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()




fig, ax = plt.subplots()
ax.scatter(Y_mean, Y_std, c=Y_K)
plt.show()



fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y_K, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Absolute Permeability (mD)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y_mean, cmap='magma', alpha=0.5)
fig.colorbar(scatter, label='Average Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y_std, cmap='viridis', alpha=0.5)
fig.colorbar(scatter, label='Standard Deviation of Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()






np.savetxt('X_raw.csv', X_raw, delimiter=',')
np.savetxt('Y_raw.csv', Y_raw, delimiter=',')
np.savetxt('T.csv', T_X, delimiter=',')







# Next time i wanna change something
# git add .                         {puts it into a queue - everything}
#   OR git add [filename]           {puts it into a queue - files specified}
# git commit -m "[description]"     {puts it together into one thing}
# git push -u origin main           {puts changes into cloud}
