import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import flopy
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap

def load_verts(fname):
    return(np.genfromtxt(fname))

def load_iverts(fname):
    f = open(fname, 'r')
    iverts = []
    xc = []
    yc = []
    for line in f:
        ll = line.strip().split()
        iverts.append([int(i) - 1 for i in ll[4:]])
        xc.append(float(ll[1]))
        yc.append(float(ll[2]))
    return iverts, np.array(xc), np.array(yc)


print(sys.version)
print('numpy version: {}'.format(np.__version__))
# print('matplotlib version: {}'.format(mpl.__version__))
print('flopy version: {}'.format(flopy.__version__))

# This is a folder containing some unstructured grids
absolute_path   = os.path.dirname(__file__)
datapth = os.path.join(absolute_path, 'data', 'unstructured')

# Load your simulation and get the head data
# ... (same code as before)
# load vertices
fname = os.path.join("/home/janek/Documents/Python/NeckarDISU/data/unstructured", 'ugrid_verts.dat')
verts = load_verts(fname)[:, 1:]

# load the incidence list into iverts
fname = os.path.join(datapth, 'ugrid_iverts.dat')
iverts, xc, yc = load_iverts(fname)

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

head                            = model.output.head().get_data()
head_values                     = head.flatten()
head_values[head_values > 500] = np.nan

# Calculate the minimum and maximum head values
valid_head_values = head_values[~np.isnan(head_values)]
min_head = np.min(valid_head_values)
max_head = np.max(valid_head_values)
# Create a colormap
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'green', 'red'])

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate through each cell defined by iverts and plot the corresponding rectangle
for cell_verts, head in zip(iverts, head_values):
    if len(cell_verts) >= 3:
        cell_x = verts[cell_verts, 0]
        cell_y = verts[cell_verts, 1]

        # Calculate the coordinates of the bottom-left corner of the cell
        x_min = np.min(cell_x)
        y_min = np.min(cell_y)
        
        # Calculate the width and height of the cell
        width = np.max(cell_x) - x_min
        height = np.max(cell_y) - y_min

        # Create a Rectangle patch and add it to the plot with color based on head value
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=1, edgecolor='none', facecolor=cmap(head)
        )
        ax.add_patch(rect)

# Set colorbar
sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_clim(min_head, max_head)
plt.colorbar(sm, label='Head Value')

# Set axis limits and display the plot
ax.set_xlim(np.min(verts[:, 0]), np.max(verts[:, 0]))
ax.set_ylim(np.min(verts[:, 1]), np.max(verts[:, 1]))
ax.set_aspect('equal', 'box')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Grid of Cells with Colored Patches')
plt.grid(True)
plt.show()


# # Create a figure and axis
# fig, ax = plt.subplots()

# # Iterate through each cell defined by iverts and plot the corresponding rectangle
# for cell_verts in iverts:
#     if len(cell_verts) >= 3:
#         cell_x = verts[cell_verts, 0]
#         cell_y = verts[cell_verts, 1]

#         # Calculate the coordinates of the bottom-left corner of the cell
#         x_min = np.min(cell_x)
#         y_min = np.min(cell_y)
        
#         # Calculate the width and height of the cell
#         width = np.max(cell_x) - x_min
#         height = np.max(cell_y) - y_min

#         # Create a Rectangle patch and add it to the plot
#         rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='black', facecolor='none')
#         ax.add_patch(rect)

# # Set axis limits and display the plot
# ax.set_xlim(np.min(verts[:, 0]), np.max(verts[:, 0]))
# ax.set_ylim(np.min(verts[:, 1]), np.max(verts[:, 1]))
# ax.set_aspect('equal', 'box')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Grid of Cells')
# plt.grid(True)
# plt.show()









