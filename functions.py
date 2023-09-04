import numpy as np
# import matplotlib.pyplot as plt
# from gstools import krige
import shutil
from gstools import Matern
# from generator import generator
# from joblib import Parallel, delayed
# import os
from pykrige.ok import OrdinaryKriging

def Directories(from_directory, absolute_path, inter_path, i):
    
    ens_dir      = absolute_path + inter_path + "/ensemble/m" + str(i)
    ens_hds_dir  = absolute_path + inter_path + "/ensemble/m" + str(i) \
                                    + "/flow_output/flow.hds"
    shutil.copytree(from_directory, ens_dir)
    
    return ens_dir, ens_hds_dir

def covmod(var, lx, ang, dim = 2):
    
    cov_mod = Matern(dim = 2, var = var, len_scale = lx,  angles=ang)
    
    return cov_mod


def kriggle(pack,porg = "points", pert = False):
    
    # Unpack Data
    x,y,data,cov_mod    = pack
    xyk                 = data[0]
    k                   = data[1]
    
    if pert == True:
        k_pert = np.random.normal(k,0.1*k)
        ok1 = OrdinaryKriging(xyk[:,0], xyk[:,1], k_pert, cov_mod)
        z1,_ = ok1.execute(porg, x, y)
        
        return k_pert, np.exp(z1)
    else:
        ok1 = OrdinaryKriging(xyk[:,0], xyk[:,1], k, cov_mod)
        z1,_ = ok1.execute(porg, x, y)
        
        return np.exp(z1)

def ole(obs, obs_sim, sigma, ens_n):
    # NRMSE of the observation locations
    nt = obs.shape[0]  
    nc = obs.shape[1]
    er = np.sqrt(1/(nt*nc) * np.sum(np.sum((obs - obs_sim)**2 / sigma)))
    
    return er      


# def Kriging(cov_mod, PP_pos, Zone_K, X, Y):
    
#     krig = krige.Ordinary(cov_mod, cond_pos=[PP_pos[:,0],PP_pos[:,1]], cond_val= Zone_K)
#     field = krig([X,Y])
#     k_res = np.reshape(field[0],(100,250))
    
#     return k_res

# def k_gen(nx, dx, lx, ang, sigma2,  mu, PP_cell):

#     k = generator(nx, dx, lx, ang, sigma2, mu)
#     Zone_K  = np.ones((len(PP_cell),1))
    
#     for j in range(len(PP_cell)):
#         Zone_K[j,0] = k[PP_cell[j][0], PP_cell[j][1]]
        
#     return k, Zone_K



# def updateK(ens_n, Zone_K, Xnew, PP_pos, n_cores, cov_mod, X, Y, k_ens):
    
#     # Update K_fields at Pilot Points
#     for j in range(ens_n):
#         Zone_K[:,j+1] = Xnew[0:len(PP_pos),j]
    
#     # Create new K-fields with Kriging
#     result = Parallel(
#         n_jobs=n_cores)(delayed(Kriging)(
#             cov_mod, PP_pos, Zone_K[:,j+1], X, Y) for j in range(ens_n)
#             )
    
#     # Assigning solutions to ensemble members
#     for j in range(ens_n-1):
#         k_ens[j+1] = np.squeeze(result[j])
        
#     return k_ens

    
# def plotOLE(n_obs, obs_true, obs_sim, t_enkf, sigma, dir_ole):
    
#     for i in range(n_obs):   
#         nt = 365 
#         nc = 1
#         er = np.sqrt(1/(nt*nc) * np.sum(np.sum((obs_true[:,i] - obs_sim[:,i])**2 / sigma)))
#         plt.figure()
#         plt.title("Truth vs. Simulated at obs. " + str(int(i+1)) + ' with OLE = ' + str("%.2f" % er))
#         plt.plot(obs_true[:,i], label = 'Truth')
#         plt.plot(obs_sim[:,i], label = 'Simulated')
#         plt.legend()
#         plt.vlines(t_enkf, 
#                    np.min((np.min(obs_true[:,i]),np.min(obs_true[:,i]))), 
#                    np.max((np.max(obs_true[:,i]),np.max(obs_sim[:,i]))), 
#                    colors='k', 
#                    linestyles='solid')
#         plt.savefig(dir_ole  + str(int(i+1)) + '.png')

# def isPerfectSquare(x):
#     s = int(np.sqrt(x))
#     return s*s == x

# def isFibonacci(n):
 
#     # n is Fibonacci if one of 5*n*n + 4 or 5*n*n - 4 or both
#     # is a perfect square
#     return isPerfectSquare(5*n*n + 4) or isPerfectSquare(5*n*n - 4)

# def saveRes(ole, te1, te2, true_heads, K_true, Ens_h_mean_mat, Ens_h_var_mat,
#             Ens_lnK_mean_mat, Ens_lnK_var_mat, name, n_obs, Y_sim,
#             tstp, t_enkf, sigma, obsYcell, obsXcell):
    
#     # Print errors to result folder
#     dir_res = 'Results/'+str(name)
#     dir_ole = dir_res + '/OLE/'
#     os.mkdir(dir_res)
#     os.mkdir(dir_ole)
#     with open(dir_res +'/Errors.txt', 'w') as f:
#         f.write('OLE' + str(ole) + '\n TE1' + str(te1) +'\n TE2' + str(te2))
#         ###CONTINUE HERE
#     with open(dir_res +'/Geometry.txt', 'w') as f:
#         f.write('OLE' + str(ole) + '\n TE1' + str(te1) +'\n TE2' + str(te2))
     
#     obs_sim = np.mean(Y_sim, axis = 2)
#     obs_true = np.ones(np.shape(obs_sim))

#     for i in range(tstp):
#         for j in range(n_obs):
#             obs_true[i,j] = true_heads[i,int(obsYcell[j]),int(obsXcell[j])]
#     plotOLE(n_obs, obs_true, obs_sim, t_enkf, sigma, dir_ole)
        