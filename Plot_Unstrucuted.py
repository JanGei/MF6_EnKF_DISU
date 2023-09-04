import matplotlib.pyplot as plt
import os
import sys
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import flopy
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.tri as mtri

# Simple functions to load vertices and incidence lists
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

def plot_head_distribution(xc, yc, iverts, reshaped_head):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a list of polygons for each cell
    polygons = []
    for cell_verts in iverts:
        if len(cell_verts) >= 3:  # Ensure there are at least 3 vertices for a cell
            cell_coords = np.array([[xc[i], yc[i]] for i in cell_verts])
            polygon = patches.Polygon(cell_coords, closed=True)
            polygons.append(polygon)

    # Create a PatchCollection with the polygons and assign head values
    collection = PatchCollection(polygons, cmap='viridis', edgecolor='0.5')
    collection.set_array(reshaped_head)

    # Add the PatchCollection to the axis
    ax.add_collection(collection)

    # Add colorbar
    cbar = plt.colorbar(collection, ax=ax, label='Head')

    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Show the plot
    plt.title('Head Distribution')
    plt.show()


print(sys.version)
print('numpy version: {}'.format(np.__version__))
print('matplotlib version: {}'.format(mpl.__version__))
print('flopy version: {}'.format(flopy.__version__))

# This is a folder containing some unstructured grids
absolute_path   = os.path.dirname(__file__)
datapth = os.path.join(absolute_path, 'data', 'unstructured')



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
reshaped_head               = head.flatten()

# load vertices
fname = os.path.join("/home/janek/Documents/Python/NeckarDISU/data/unstructured", 'ugrid_verts.dat')
verts = load_verts(fname)[:, 1:]

# load the incidence list into iverts
fname = os.path.join(datapth, 'ugrid_iverts.dat')
iverts, xc, yc = load_iverts(fname)

print("Length of reshaped_head:", len(reshaped_head))
print("Length of xc:", len(xc))
print("Length of yc:", len(yc))

# Convert iverts to a list of triangles
triangles = []
for cell_verts in iverts:
    if len(cell_verts) >= 3:  # Ensure there are at least 3 vertices for a triangle
        for i in range(1, len(cell_verts) - 1):
            triangles.append([cell_verts[0], cell_verts[i], cell_verts[i + 1]])

# Check triangles
print("Number of triangles:", len(triangles))

# Create a triangulation
triang = Triangulation(xc, yc, triangles=triangles)

reshaped_head[reshaped_head > 500] = np.nan

# Create a scatter plot with the head values
# plt.figure()
# plt.tripcolor(triang, reshaped_head, shading='flat', cmap='viridis')
# plt.colorbar(label='Head')
# plt.gca().set_aspect('equal')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Head Distribution')
# plt.show()



# Call the plotting function
plot_head_distribution(xc, yc, iverts, reshaped_head)

# mm = flopy.plot.map.PlotMapView(model=model)
# mm.plot_array(reshaped_head, edgecolor='0.5')
# mm.plot_grid()
# cs = mm.contour_array(head)
# mm.ax.clabel(cs)
# # mm.plot_vector(qx, qy, normalize=True)
# plt.show()

# Simple functions to load vertices and incidence lists
# def load_verts(fname):
#     return(np.genfromtxt(fname))

# def load_iverts(fname):
#     f = open(fname, 'r')
#     iverts = []
#     xc = []
#     yc = []
#     for line in f:
#         ll = line.strip().split()
#         iverts.append([int(i) - 1 for i in ll[4:]])
#         xc.append(float(ll[1]))
#         yc.append(float(ll[2]))
#     return iverts, np.array(xc), np.array(yc)

# # load vertices
# fname = os.path.join("/home/janek/Documents/Python/NeckarDISU/data/unstructured", 'ugrid_verts.dat')
# verts = load_verts(fname)[:, 1:]

# # load the incidence list into iverts
# fname = os.path.join(datapth, 'ugrid_iverts.dat')
# iverts, xc, yc = load_iverts(fname)

# # Print the first 5 entries in verts and iverts
# for ivert, v in enumerate(verts[:5]):
#     print('Vertex coordinate pair for vertex {}: {}'.format(ivert, v))
# print('...\n')
    
# for icell, vertlist in enumerate(iverts[:5]):
#     print('List of vertices for cell {}: {}'.format(icell, vertlist))
    
# ncpl = np.array(5 * [len(iverts)])
# sr = flopy.utils.reference.SpatialReferenceUnstructured(xc, yc, verts, iverts, ncpl)
# print(ncpl)
# print(sr)

# print("Length of xc:", len(xc))
# print("Length of yc:", len(yc))
# print("Length of verts:", len(verts))
# print("Length of iverts:", len(iverts))


