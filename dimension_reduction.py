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


num_x = 5
num_y = 5
num_z = 5

spacing = 1e-4
pn = op.network.Demo(shape=[num_x, num_y, num_z], spacing=spacing)
air = op.phase.Air(network=pn)
N_X = 425

N_train_val = 1000
N_test = 10

X = np.zeros((N_train_val + N_test, N_X))
Y = np.zeros((N_train_val + N_test, 1))
Y_mean = np.zeros((N_train_val + N_test, 1))
Y_std = np.zeros((N_train_val + N_test, 1))
Y_min = np.zeros((N_train_val + N_test, 1))
Y_max = np.zeros((N_train_val + N_test, 1))
Y_por = np.zeros((N_train_val + N_test, 1))



    

for i in range(N_train_val + N_test):
    print(i)

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
                  diameter='throat.diameter',)
    
    pn['pore.volume'] = 0.0
    
    drn = op.algorithms.Drainage(network=pn, phase=air)
    drn.set_inlet_BC(pores=pn.pores('left'))
    drn.run()
    data = drn.pc_curve()
    
    ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
    ip.set_inlet_BC(pores=pn.pores('left'))
    ip.run()
    data_ip = ip.pc_curve()
    
    # plt.figure(dpi=300)
    # # plt.step(data.pc, data.snwp, 'b--', where='post')
    # plt.plot(data_ip.pc, data_ip.snwp, c='green', linewidth=2)
    # plt.xlabel('Capillary Pressure')
    # plt.xlim([0, 20000])
    # plt.ylabel('Non-Wetting Phase Saturation')
    # plt.ylim([0, 1])
    # plt.show()
    
    
    
    
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
    # A = spacing*num_y*spacing*num_z #np.max(pn['pore.coords'][:,1])*np.max(pn['pore.coords'][:,2])
    # L = spacing*num_x #np.max(pn['pore.coords'][:,0])
    K = Q * L / A
    
    K_mD = K/0.98e-12*1000
    print('Permeability = ', K_mD, ' mD')
    
    

    X[i] = data_ip.pc
    Y[i] = K_mD
    
    Y_mean[i] = np.mean(pn['throat.diameter'])
    Y_std[i] = np.std(pn['throat.diameter'])
    Y_min[i] = np.min(pn['throat.diameter'])
    Y_max[i] = np.max(pn['throat.diameter'])
    Y_por[i] = np.sum(pn['throat.diameter']**2*3.14*pn['throat.length']/(4))/(A*L)
    
    
    
    
    
    # coords = pn['pore.coords']
    # conns = pn['throat.conns']
    # td = pn['throat.diameter']
    
    # segments = []
    # for t in range(len(conns)):
    #     p1, p2 = conns[t]
    #     segments.append([
    #         coords[p1],
    #         coords[p2]
    #     ])
    # segments = np.array(segments)
    
    # norm = mcolors.Normalize(vmin=td.min(), vmax=td.max())
    # cmap = cm.brg
    
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111, projection='3d')
    
    # lc = Line3DCollection(
    #     segments,
    #     cmap=cmap,
    #     norm=norm,
    #     linewidths= 0.5 + 5*(td - td.min()) / (td.max() - td.min()),
    #     alpha=0.8
    # )
    # lc.set_array(td)
    # ax.add_collection3d(lc)
    
    # ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    # ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    # ax.set_zlim(coords[:, 2].min(), coords[:, 2].max())
    # ax.set_box_aspect([1, 1, 1])
    # # ax.set_axis_off()
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    
    # cbar = fig.colorbar(lc, ax=ax, pad=0.1, shrink=0.7)
    # cbar.set_label('Throat Diameter (m)')
    
    # plt.tight_layout()
    # plt.show()


    


X = X - np.mean(X, axis=0)

Sigma = X.T @ X
lam, W = np.linalg.eigh(Sigma)
idx = lam.argsort()[::-1]
lam = lam[idx]
W = W[:, idx]



info = lam/np.sum(lam)
cum = np.cumsum(info)

k = 3

print(len(lam[cum < 0.9]))


T_X = X @ W[:,:k]

op.visualization.set_mpl_style()



fig, ax = plt.subplots(3, 1, dpi=300, figsize=(5, 6))
ax[0].hist(Y, color='pink', bins=20)
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
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y, cmap='jet', alpha=0.5)
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

fig, ax = plt.subplots(dpi=300)
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y_min, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Minimum Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()

fig, ax = plt.subplots(dpi=300)
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y_max, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Maximum Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()

fig, ax = plt.subplots(dpi=300)
scatter = ax.scatter(T_X[:,0], T_X[:,1], c=Y_por, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Porosity')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.show()




fig, ax = plt.subplots()
ax.scatter(Y_mean, Y_std, c=Y)
plt.show()



fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y, cmap='jet', alpha=0.5)
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

fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y_min, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Minimum Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y_max, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Maximum Throat Size (m)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()


fig = plt.figure(dpi=300, figsize=(15, 4))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(T_X[:,0], T_X[:,1], T_X[:,2], c=Y_por, cmap='jet', alpha=0.5)
fig.colorbar(scatter, label='Porosity')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
