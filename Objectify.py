import numpy as np
import flopy 
from joblib import Parallel, delayed
# from gstools import krige, Matern
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

class Ensemble:
    def __init__(self, X, Ysim, obsloc, PPcor, tstp, meanh, meank, ncores):
        self.ncores = ncores
        self.X      = X
        self.nreal  = X.shape[1]
        self.Ysim   = Ysim
        self.nobs   = Ysim.shape[0] #CHECK whetehr this is true
        self.obsloc = obsloc
        self.PPcor  = PPcor
        self.nPP    = len(PPcor)
        self.tstp   = tstp
        self.meanh  = meanh
        self.meank  = meank
        self.members = []
        
    def update_tstp(self):
        self.tstp += 1 
        
    # Add Member to Ensemble
    def add_member(self, member):
        self.members.append(member)
        
    def remove_member(self, member, j):
        self.members.remove(member)
        np.delete(self.X,       j, axis = 1)
        np.delete(self.Ysim,    j, axis = 1)
        self.nreal -= 1
        
    def set_transient_forcing(self, Rch, Qpu):
        # Set transient forcings and create appropriate dictionaries
        rch                     = self.members[0].model.get_package("rch")
        rchspd                  = rch.stress_period_data.get_data()
        rchspd[0]['recharge']   = Rch
                
        wel      = self.members[0].model.get_package("wel")
        wspd     = wel.stress_period_data.get_data()
        
        for entry in Qpu.keys():
            if entry in wspd[0]['boundname']:
                # This is not the correct way of accessing a dictionary
                wspd[0]['q'][np.where(wspd[0]['boundname'] == entry)[0][0]] = -Qpu[entry][self.tstp]
        
        for member in self.members:
            member.model.rch.stress_period_data.set_data(rchspd)
            member.model.wel.stress_period_data.set_data(wspd)
    
    def initial_conditions(self):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                ) for member in self.members
                )
        # self.ncores
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                # UPDATE INITIAL CONDITIONS
                self.members[j].set_hfield(np.squeeze(result[j]))
                        
                j = j + 1
        
    # Propagate Entire Ensemble
    def predict(self):
        
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                ) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                self.X[self.nPP:,j]  = np.ndarray.flatten(result[j])
                
                # THESE VALUES SHOULD BE DIFFERENT --> CHECK Kfield maybe?
                
                # k = 0
                # for key in self.obsloc.keys():
                #     self.Ysim[k,j] = self.X[self.nPP+int(self.obsloc[key]), j]
                #     k +=1
                    # self.Ysim[k,j]  = self.members[j].model.output.head().get_data()[0,self.obsloc[k][0],self.obsloc[k][1]]

                        
                j +=  1
    def updateObservations(self,Obs_t):
        Ysim            = np.zeros((len(Obs_t),self.nreal))
        
        for i in range(self.nreal):
            for k in range(len(Obs_t)):
                Ysim[k,i] = self.X[self.nPP+int(Obs_t[k,0]), i]
                
        self.Ysim = Ysim
    
    def update_hmean(self):
        newmean = np.zeros(self.members[0].model.output.head().get_data().shape)
        for member in self.members:
            newmean += member.model.ic.strt.array
        self.meanh = newmean / self.nreal
        
    def update_kmean(self):
        newmean = np.zeros(self.members[0].model.npf.k.array.shape)
        for member in self.members:
            newmean += member.model.npf.k.array
        self.meank = newmean / self.nreal
        
    def get_varh(self):
        return  np.reshape(np.var(self.X[self.nPP:], axis = 1), self.members[0].model.npf.k.array.shape)    
    
    def update_PP(self, PPk):
        for j in range(self.nreal):
            self.X[0:self.nPP, j] = PPk[:,j]
               
    def analysis(self, eps):
        
        # Compute mean of postX and Y_sim
        Xmean   = np.tile(np.array(np.mean(np.ma.masked_values(self.X, 1e+30), axis = 1)).T, (self.nreal, 1)).T
        Ymean   = np.tile(np.array(np.mean(self.Ysim,  axis  = 1)).T, (self.nreal, 1)).T
        
        # Fluctuations around mean
        X_prime = np.ma.masked_values(self.X, 1e+30) - Xmean
        Y_prime = self.Ysim  - Ymean
        
        # Variance inflation
        # priorX  = X_prime * 1.01 + Xmean
        
        # Measurement uncertainty matrix
        R       = np.identity(self.nobs) * eps 
        
        # Covariance matrix
        Cyy     = 1/(self.nreal-1)*np.matmul((Y_prime),(Y_prime).T) + R 
                        
        return X_prime, Y_prime, Cyy
    
    def Kalman_update(self, damp, X_prime, Y_prime, Cyy, Obs_t):
        
        Y_obs = np.tile(Obs_t[:,1],(self.nreal,1)).transpose()
        
        self.X += 1/(self.nreal-1) * (damp *
                    np.matmul(
                        np.ma.getdata(X_prime), np.matmul(
                            Y_prime.T, np.matmul(
                                np.linalg.inv(Cyy), (Y_obs - self.Ysim) #This line seems to be the problem
                                )
                            )
                        ).T
                    ).T
        
        for j in range(len(self.members)):
            self.members[j].set_hfield(np.reshape(self.X[self.nPP:,j],self.members[j].model.ic.strt.array.shape))
        
        self.update_hmean()
        
    def updateK(self, pack):
        
        x,y,xyk,k,cov_mod    = pack

        Parallel(n_jobs=self.ncores)(delayed(self.members[j].updateK)(
                [x,y,xyk,self.X[0:self.nPP,j],cov_mod]) for j in range(len(self.members))
                )
        
        self.update_kmean()
        
    def get_obs(self):
        obs = [self.meanh[0, self.obsloc[i][0], self.obsloc[i][1]] for i in range(len(self.obsloc))]
        return obs
    
    def ole(self, interim, Ole_mat, sigma, Errorcounter):
        
        active_obs  = interim.drop(['Date'], axis = 1)[interim.drop(['Date'], axis =1) > 0].count().sum()
        obs         = np.zeros((1,active_obs))
        obs_sim     = np.zeros((1,active_obs))
        
        # Observation location error, n_nodes up until that timestep, unscaled NRMSE up until that point
        counter = 0
        for key in self.obsloc.keys():
            if interim[key][interim[key].index[0]] > 0:
                obs[0,counter]      = interim[key][interim[key].index[0]]
                obs_sim[0,counter]  = self.meanh[0,0,int(self.obsloc[key])]
                counter += 1
       
        if Errorcounter == 0:
            uNMRSE  = np.sum(np.sum((obs - obs_sim)**2 / sigma))
            n_node  = active_obs
            ole     = np.sqrt(1/n_node * uNMRSE)       
        else:
            uNMRSE  = np.sum(np.sum((obs - obs_sim)**2 / sigma)) + Ole_mat[Errorcounter-1, 2]
            n_node  = active_obs + Ole_mat[Errorcounter-1, 1]
            ole     = np.sqrt(1/n_node * uNMRSE) 
            
        return np.array([ole, n_node, uNMRSE])
    
    def compare(self):
        
        result  = Parallel(n_jobs=self.ncores)(delayed(self.members[j].get_spdis)
                                              () for j in range(len(self.members))
                )
        
        qx_l = [result[i][0] for i in range(len(result))]
        qy_l = [result[i][1] for i in range(len(result))]
        qz_l = [result[i][2] for i in range(len(result))]
        
        qx = np.squeeze(np.mean(qx_l, axis=0))
        qy = np.squeeze(np.mean(qy_l, axis=0))
        qz = np.squeeze(np.mean(qz_l, axis = 0))
        
        mask      = self.meanh > 1e+10
        
        vmin            = self.meanh[self.meanh > -0.1].min()
        vmax            = self.meanh[self.meanh < 500].max()
        
        k_field                         = self.meank
        k_field                         = np.log10(np.array(k_field)/86400)

        k_field[np.squeeze(mask)]       = 1e+30
                
        vmink                           = k_field.min()
        vmaxk                           = k_field[k_field < 86400].max()
        
        fig1, axes1 = plt.subplots(1, 1, figsize=(25, 25), sharex=True, dpi = 400)
        ax11 = axes1

        ax11.set_title("Ensemble-mean K-field in period " + str(int(self.tstp)), fontsize = 30)
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax11)
        ax11.set_aspect("equal")
        # mapable = pmv.plot_array(k_field.flatten(), cmap="RdBu", vmin=vmink, vmax=vmaxk)
        mapable = pmv.plot_array(self.members[0].model.npf.k.array, cmap="RdBu", vmin=vmink, vmax=vmaxk)
        # im_ratio = k_field.shape[1]/k_field.shape[2]
        cbar = plt.colorbar(mapable, pad=0.04, ax =ax11)
        # cbar = plt.colorbar(mapable, fraction=im_ratio*0.046, pad=0.04, ax =ax11)
        cbar.ax.set_ylabel('Hydraulic Conductivity [m/s]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax11.yaxis.label.set_size(25)
        ax11.xaxis.label.set_size(25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax11.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        pmv.plot_vector(qx, qy, width=0.0008, color="black")
        
        plt.savefig("K_field in t" + str(self.tstp), format="svg")
        
        fig2, axes2 = plt.subplots(1, 1, figsize=(25, 25), sharex=True, dpi = 400)
        ax21 = axes2

        ax21.set_title("Ensemble-mean h-field in period " + str(int(self.tstp)), fontsize = 30)
        pmv = flopy.plot.PlotMapView(self.members[0].model, layer=0, ax=ax21)
        ax21.set_aspect("equal")
        pmv.contour_array(
                    self.meanh, masked_values = 1e+30, levels=np.arange(vmin, vmax, 0.1), linewidths=2.0, vmin=vmin, vmax=vmax
                )
        mapable = pmv.plot_array(self.meanh, cmap="RdBu", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(mapable, pad=0.04, ax =ax21)
        cbar.ax.set_ylabel('Hydraulic Head [m]', fontsize = 25)
        cbar.ax.tick_params(labelsize=20)
        ax21.yaxis.label.set_size(25)
        ax21.xaxis.label.set_size(25)
        plt.ylabel('Northing [m]', fontsize = 25)
        ax21.tick_params(axis='both', which='major', labelsize=20)
        pmv.plot_grid(colors="k", alpha=0.1)
        pmv.plot_bc('riv')
        
        plt.savefig("Heads in t" + str(self.tstp), format="svg")
        
       
    
class Member:
        
    def __init__(self, direc, Kf, idx_ge):
        self.direc      = direc
        self.hdirec     = direc + "/flow_output/flow.hds"
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
                                verbosity_level     = 0
                                )
        self.model      = self.sim.get_model()
        self.greateq    = idx_ge
        
        self.set_kfield(Kf)
    
    def get_hfield(self):
        return self.model.output.head().get_data()
    
    def get_kfield(self):
        return self.model.npf.k.array
        
    def set_kfield(self, Kf):
        k = self.model.npf.k.get_data()
        # Change K alues for second layer
        k[10402:21058] = Kf
        k[self.greateq] = 86400
        self.model.npf.k.set_data(k)
            
    def set_hfield(self, Hf):
        self.model.ic.strt.set_data(Hf)
        
    def predict(self):
          
        success, buff = self.sim.run_simulation()
        
        if not success:
            print(f"Model in {self.direc} has failed")
            Hf = None   
        else:
            Hf = self.model.output.head().get_data()
        return Hf
    
    def updateK(self, pack, porg = "points"):
        
        x,y,xyk,k,cov_mod    = pack
        
        ok1 = OrdinaryKriging(xyk[:,0], xyk[:,1], k, cov_mod)
        z1,_ = ok1.execute(porg, x, y)
        
        self.set_kfield(np.exp(z1))

    def get_spdis(self):
        head                        = self.model.output.head().get_data()
        bud                         = self.model.output.budget()
        spdis                       = bud.get_data(text="DATA-SPDIS")[0]
        
        # maybe we should flatten everything?? so that entire array is not processeed but its values
        qx                          = np.zeros(np.shape(head))
        qy                          = np.zeros(np.shape(head))
        qz                          = np.zeros(np.shape(head))
        
        counter = 0
        for i in range(self.model.modelgrid.nnodes):
            if head[0][0][i] < 1e+30:
                qx[0][0][i]   = spdis["qx"][counter]
                qy[0][0][i]   = spdis["qy"][counter]
                qz[0][0][i]   = spdis["qz"][counter]
                counter     += 1
            
        return qx, qy, qz
    