import numpy as np
import flopy 
from joblib import Parallel, delayed
from pykrige.ok import OrdinaryKriging

class Ensemble:
    def __init__(self, X, Ysim, obscell, PPcor, htable, hvartable, tstp, meanh, varh, meank, Ens_PP, ncores):
        self.ncores = ncores
        self.X      = X
        self.nreal  = X.shape[1]
        self.nX     = X.shape[0]
        self.Ysim   = Ysim
        self.obscell= obscell
        self.PPcor  = PPcor
        self.htable = htable
        self.hvartab= hvartable
        self.tstp   = tstp
        self.meanh  = meanh
        self.varh   = varh
        self.meank  = meank
        self.Ens_PP = Ens_PP
        self.nPP    = Ens_PP.shape[0] #CHECK whetehr this is true
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
        np.delete(self.Ens_PP,  j, axis = 1)
        self.nreal -= 1
    
    def initial_conditions(self, Rch, Qpu):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                Rch, Qpu, self.tstp) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                self.members[j].set_hfield(np.squeeze(result[j]))
                        
                j = j + 1
        
    # Propagate Entire Ensemble
    def predict(self, Rch, Qpu):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                Rch, Qpu, self.tstp) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                # htable refers to the mean head level in the ensemble
                #self.htable[j,:,:]   = np.squeeze(result[j])
                self.X[self.nPP:,j]  = np.ndarray.flatten(result[j])
                        
                j +=  1
    
    def update_hmean(self):
        newmean = np.zeros(self.meanh.shape)
        for member in self.members:
            newmean += member.hfield
        self.meanh = newmean / self.nreal
        self.htable[self.tstp,:] = self.meanh
        
    def update_kmean(self):
        newmean = np.zeros(self.members[0].kfield.shape)
        for member in self.members:
            newmean += member.kfield
        self.meank[10402:21058] = newmean / self.nreal
        
    def update_hvar(self):
        self.varh = np.reshape(np.var(self.X[self.nPP:],axis = 1),self.meanh.shape)
        self.hvartab[self.tstp-1,:] = self.varh
    
    
    def update_PP(self, PPk):
        for j in range(self.nreal):
            self.X[0:self.nPP, j] = PPk[:,j]
               
    def analysis(self, Obs_t, eps):
        
        self.Ysim               = np.zeros((len(Obs_t),self.nreal))
        for j in range(self.nreal):
            for k in range(len(Obs_t)):
                self.Ysim[k,j]  = self.members[j].hfield[int(Obs_t[k,0])]
                # self.Ysim[k,j]  = self.htable[j,self.obsloc[k][0],self.obsloc[k][1]]
                
                
        # Compute mean of postX and Y_sim
        Xmean   = np.tile(np.array(np.mean(self.X, axis = 1)).T, (self.nreal, 1)).T
        Ymean   = np.tile(np.array(np.mean(self.Ysim,  axis  = 1)).T, (self.nreal, 1)).T
        
        # Fluctuations around mean
        X_prime = self.X - Xmean
        Y_prime = self.Ysim  - Ymean
        
        # Variance inflation
        # priorX  = X_prime * 1.01 + Xmean
        
        # Measurement uncertainty matrix
        R       = np.identity(len(Obs_t)) * eps 
        
        # Covariance matrix
        Cyy     = 1/(self.nreal-1)*np.matmul((Y_prime),(Y_prime).T) + R 
                        
        return X_prime, Y_prime, Cyy
    
    def Kalman_update(self, damp, X_prime, Y_prime, Cyy, Y_obs):
        
        self.X += 1/(self.nreal-1) * (damp *
                    np.matmul(
                        X_prime, np.matmul(
                            Y_prime.T, np.matmul(
                                np.linalg.inv(Cyy), (Y_obs[:,1].reshape((len(self.Ysim),1)) - self.Ysim)
                                )
                            )
                        ).T
                    ).T
        
        for j in range(len(self.members)):
            self.members[j].set_hfield(np.reshape(self.X[self.nPP:,j],self.members[j].hfield.shape))
        
        self.update_hmean()
        self.update_hvar()
        
    def updateK(self, cellx, celly, ang, lx, cov_mod):
        
        Parallel(n_jobs=self.ncores)(delayed(self.members[j].updateK)(
                [cellx,celly,[self.PPcor,self.X[0:self.nPP,j]],ang,lx,cov_mod]) for j in range(len(self.members))
                )
        
        self.update_kmean()
        
       
    
class Member:
        
    def __init__(self, direc,  kfield, hfield, mname, idx_ge):
        self.direc      = direc
        self.hdirec     = direc + "/flow_output/flow.hds"
        self.kfield     = kfield
        self.hfield     = hfield
        self.mname      = mname
        self.greateq    = idx_ge
        # This takes comparably long
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                mname, 
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
                                verbosity_level     = 0
                                )
    
    def get_hfield(self):
        return self.hfield
    
    def get_kfield(self):
        return self.kfield
        
    def set_kfield(self, Kf):
        assert Kf.shape == self.kfield.shape, "Why you change size of field?"
        
        self.kfield = Kf
        
        # Update Package
        mdl     = self.sim.get_model(self.mname)
        npf     = mdl.get_package("npf")
        
        k = npf.k.get_data()
        # Change K alues for second layer
        k[10402:21058] = np.exp(Kf)
        k[self.greateq] = 86400
        npf.k.set_data(k)
            
    def set_hfield(self, Hf):
        assert Hf.shape == self.hfield.shape, "Why you change size of field?"
        
        self.hfield = Hf
        
        # Update Package
        mdl     = self.sim.get_model(self.mname)
        ic      = mdl.get_package("ic")
         
        ic.data_list[0].set_data(Hf)
        
    def predict(self, Rch, Qpu, tstp):
        
        mdl     = self.sim.get_model(self.mname)
        
        rch                     = mdl.get_package("rch")
        rchspd                  = rch.stress_period_data.get_data()
        rchspd[0]['recharge']   = Rch
        
        rch.stress_period_data.set_data(rchspd)
        
        wel      = mdl.get_package("wel")
        wspd     = wel.stress_period_data.get_data()
        
        # for entry in wspd:
        #     if "brunnen " in entry['boundname']:
        #         wspd['q'][wspd['boundname'].index(value)] = Qpu[0]
                
        for entry in Qpu.keys():
            if entry in wspd[0]['boundname']:
                wspd[0]['q'][np.where(wspd[0]['boundname'] == entry)[0][0]] = -Qpu[entry][tstp]
        
        wel.stress_period_data.set_data(wspd)
    
        success, buff = self.sim.run_simulation()
        
        if not success:
            print(f"Model in {self.direc} has failed")
            Hf = None
               
        else:
            Hf = flopy.utils.binaryfile.HeadFile(self.hdirec).get_data(kstpkper=(0, 0))
            # self.set_hfield(self, Hf)
            
        return Hf
    
    def updateK(self, pack,porg = "points", pert = False):
        
        x,y,data,ang,lx,cov_mod = pack
        xyk = data[0]
        k = data[1]
        
        if pert == True:
            k_pert = np.random.normal(k,0.1*k)
            ok1 = OrdinaryKriging(xyk[:,0], xyk[:,1], k_pert, cov_mod)
            z1,_ = ok1.execute(porg, x, y)
        else:
            ok1 = OrdinaryKriging(xyk[:,0], xyk[:,1], k, cov_mod)
            z1,_ = ok1.execute(porg, x, y)
              

        self.set_kfield(z1)

    