import time
import os
import flopy
import shutil
import multiprocessing
import csv 
import datetime

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from numpy import genfromtxt
from scipy.io import savemat

# from scipy.io import savemat

from functions import Directories, kriggle, covmod
from Objectify import Ensemble, Member

absolute_path   = os.path.dirname(__file__)
inter_path      = "/NeckartalModel1718/NeckartalCalib_try_models/MODFLOW 6"
ens_path        = "/NeckartalModel1718/NeckartalCalib_try_models/MODFLOW 6/ensemble"
model_path      = absolute_path + inter_path + "/sim"

sim_orig        = flopy.mf6.modflow.MFSimulation.load(
                        # mname, 
                        version             = 'mf6', 
                        exe_name            = 'mf6',
                        sim_ws              = model_path, 
                        verbosity_level     = 0
                        )

model     = sim_orig.get_model()
model.npf.save_specific_discharge = True

sim_orig.run_simulation()

head                        = model.output.head().get_data()
# bud                         = model.output.budget()
# spdis                       = bud.get_data(text="DATA-SPDIS")[0]

# qx                          = np.zeros(np.shape(head))
# qy                          = np.zeros(np.shape(head))
# qz                          = np.zeros(np.shape(head))

# counter = 0
# for i in range(model.modelgrid.nnodes):
#     if head[0][0][i] < 1e+30:
#         qx[0][0][i]   = spdis["qx"][counter]
#         qy[0][0][i]   = spdis["qy"][counter]
#         qz[0][0][i]   = spdis["qz"][counter]
#         counter     += 1


# qx_l = [result[i][0] for i in range(len(result))]
# qy_l = [result[i][1] for i in range(len(result))]
# qz_l = [result[i][2] for i in range(len(result))]

# qx = np.squeeze(np.mean(qx_l, axis=0))
# qy = np.squeeze(np.mean(qy_l, axis=0))
# qz = np.squeeze(np.mean(qz_l, axis = 0))

# mask      = head > 1e+10

vmin            = head[head > -0.1].min()
vmax            = head[head < 500].max()

g = model.modelgrid
disu = model.disu
a = np.zeros((disu.nodes.array), dtype = int)

# k_field                         = model.npf.k.array
# k_field                         = np.log10(np.array(k_field)/86400)

# k_field[np.squeeze(mask)]       = 1e+30
        
# vmink                           = k_field.min()
# vmaxk                           = k_field[k_field < 86400].max()

fig1, axes1 = plt.subplots(1, 1, figsize=(25, 25), sharex=True, dpi = 400)
ax11 = axes1

ax11.set_title("Ensemble-mean K-field in period " + str(int(0)), fontsize = 30)
# g.plot(ax = ax11, colors =[0.5, 0.5, 0.5])
pmv = flopy.plot.PlotMapView(model, modelgrid = g, ax=ax11)
pmv.plot_grid(ax = ax11, colors = np.array([0, 0, 0]))
# mapable = pmv.plot_array(head, cmap="RdBu", vmin=vmin, vmax=vmax)
# plt.scatter(g.xcellcenters, g.ycellcenters)



# 
# ax11.set_aspect("equal")
# # mapable = pmv.plot_array(k_field.flatten(), cmap="RdBu", vmin=vmink, vmax=vmaxk)
# mapable = pmv.plot_array(np.squeeze(head), cmap="RdBu", vmin=vmin, vmax=vmax)
# # im_ratio = k_field.shape[1]/k_field.shape[2]
# cbar = plt.colorbar(mapable, pad=0.04, ax =ax11)
# # cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax11)
# cbar.ax.set_ylabel('Hydraulic Conductivity [m/s]', fontsize = 25)
# cbar.ax.tick_params(labelsize=20)
# ax11.yaxis.label.set_size(25)
# ax11.xaxis.label.set_size(25)
# plt.ylabel('Northing [m]', fontsize = 25)
# ax11.tick_params(axis='both', which='major', labelsize=20)
# pmv.plot_grid(colors="k", alpha=0.1)
# pmv.plot_bc('riv')
# pmv.plot_vector(qx, qy, width=0.0008, color="black")