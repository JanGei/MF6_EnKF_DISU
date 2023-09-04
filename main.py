import time
import os
import flopy
import shutil
import multiprocessing
import csv 
import datetime

# THIS SCIPT IS NOT ABLE TO PLOT ANYTHING AT THE MOMENT

import numpy as np
import geopandas as gpd
import pandas as pd

from joblib import Parallel, delayed
from numpy import genfromtxt
from scipy.io import savemat

# from scipy.io import savemat

from functions import Directories, kriggle, covmod
from Objectify import Ensemble, Member

if __name__ == '__main__':
    
    # =========================================================================
    #### Defining Paths & Loading Original (To be copied) Model
    # =========================================================================
    start_time      = time.perf_counter()

    absolute_path   = os.path.dirname(__file__)
    inter_path      = "/NeckartalModel1718/NeckartalCalib_try_models/MODFLOW 6"
    ens_path        = "/NeckartalModel1718/NeckartalCalib_try_models/MODFLOW 6/ensemble"
    model_path      = absolute_path + inter_path + "/sim"
    
    shapefile       = gpd.read_file(absolute_path+"/Modellgebiet/Outline_altered_east.shp")
    
    # This Model is not to be changed during the computation
    sim_orig        = flopy.mf6.modflow.MFSimulation.load(
                            # mname, 
                            version             = 'mf6', 
                            exe_name            = 'mf6',
                            sim_ws              = model_path, 
                            verbosity_level     = 0
                            )
    
    model_orig      = sim_orig.get_model()
    
    model_orig.npf.save_specific_discharge = True
    
    # check whether model is in transient mode
    # sim_orig.tdis.nper
    # model_orig.sto.transient
    # Transient conditions will apply until the STEADY-STATE
    # keyword is specified in a subsequent BEGIN PERIOD block.
    
    # Here, preliminary changes to the model can be appllied before it is copied
    
    sim_orig.write_simulation()
    
    p_names = model_orig.package_names
    if 'npf' in p_names:
        npf     = model_orig.get_package('npf')
        k_orig  = npf.k.data
    else:
        print("How do you expect to run a MODFLOW model without an NPF package?")
        exit()

    # DISU PACKAGE - We're getting cell mid points
    if 'disu' in p_names:
        disu                    = model_orig.get_package('disu')
        n_c                     = len(disu.bot.array)
        idx                     = np.empty((n_c),dtype = bool)
        idx_ge                  = np.empty((n_c),dtype = bool)
        idx_ge[:]               = False
        idx_ge[k_orig == 86400] = True
        idx[:]                  = False
        idx[10402:21058]        = True 
        cellx, celly            = disu.cell2d.array.xc[idx], disu.cell2d.array.yc[idx]
        # What about .xcellcenters .ycellcenters .zcellcenters and .xyzcellcenters
    else:
        print("You refer to a model without the DISU package")
        exit()
        
    if 'rch' in p_names:
        rch     = model_orig.get_package('rch')
        rspd    = rch.stress_period_data.get_data()[0]
        f_rch   = rspd['recharge']/np.max(rspd['recharge'])
        
    # WEL PACKAGE -- We're getting cell IDs of wells
    if 'wel' in p_names:
        wel     = model_orig.get_package('wel')
        # wspd contains fixed flow boundaries and wells
        wspd    = wel.stress_period_data.get_data()[0]
        wellidx = []
        names = []
        # Checking for actual wells
        for entry in wspd:
            # sorting for wells - Problem auftrieb riedbrunnen - ist kein brunnen!!
            if "brunnen " in entry['boundname']:
                wellidx.append(int(entry['cellid'][0]))
                names.append(entry['boundname'])

    if 'sto' in p_names:
        sto     = model_orig.get_package('sto')
    else:
        print("How do you expect to run a TRANSIENT model without a STO package?")
        exit()
    
    
    finish_time = time.perf_counter()
    print("Original Model loaded in {} seconds - using sequential processing"
          .format(finish_time-start_time)
          )
    print("---")
    # =========================================================================
    #### Copying original Model and defining Ensemble
    # =========================================================================
    start_time      = time.perf_counter()
    
    if os.path.isdir(absolute_path + ens_path) == True :
        shutil.rmtree(absolute_path + ens_path)
    
    ncores          = multiprocessing.cpu_count()
    nreal           = 5
    
    ens_dir         = [None] * nreal
    ens_hds_dir     = [None] * nreal
    
    result = Parallel(n_jobs=ncores)(delayed(Directories)(
        model_path, absolute_path, inter_path, i) 
        for i in range(nreal)
        )
    
    for i in range(nreal):
        ens_dir[i]      = result[i][0]
        ens_hds_dir[i]  = result[i][1]
        
    finish_time = time.perf_counter()
    print("Ensemble directories created in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("---")
    # =========================================================================
    #### Transient data | Pilot Points | Observations
    # =========================================================================

    # Importing transient data - Here we certainly need API requests
    Rch_data    = genfromtxt(absolute_path  +'/csv data/RCHunterjesingen.csv', delimiter=',')
    Qpu_data    = pd.read_csv(absolute_path +'/csv data/2017.csv')
    Obs_data    = pd.read_csv(absolute_path +'/csv data/Pegel.csv', sep = ',')
    Obs_data['Date']= pd.to_datetime(Obs_data['Date'], dayfirst = True)

    # Obs_data['Date'] = Obs_data['Date'].dt.strftime('%d.%m.%Y')

    with open(absolute_path +'/csv data/Pegel_Cell_ID.csv') as f:
        next(f) 
        reader      = csv.reader(f, skipinitialspace=True)
        obs_cell    = dict(reader)

    # =========================================================================
    #### Generating K_ensemble -- with values at Pilotpoints and their stdevs
    # =========================================================================
    
    start_time = time.perf_counter()

    # import pilot points / pumping tests
    pv      = genfromtxt(absolute_path + '/csv data/PV_pseudoPP9_2.csv', dtype = str, delimiter=",")
    xyPP    = pv[1:,1:3].astype(float)
    k_avg   = np.log(pv[1:,9].astype(float))
    k_min   = np.log(pv[1:,7].astype(float))
    k_max   = np.log(pv[1:,8].astype(float))
    k_cal   = np.log(pv[1:,11].astype(float))
    group   = pv[1:,6].astype(int)
    n_PP    = len(xyPP)

    # Freddi hat nur layer2 gekriggt - Identifier finden von den Indizes?

    # log(Mean), log(variance), corellation lengths, angle
    lx      = np.array([1000,1100])
    ang     = 24
    var     = 1.5
    
    cov_mod = covmod(var, lx, ang)
    
    data    = [xyPP,k_cal]
    krig_p  = [cellx,celly,data,cov_mod]

    # Generating initial K-fields 
    result = Parallel(n_jobs=ncores)(delayed(kriggle)(
        krig_p, pert = True) 
        for i in range(nreal)
        )

    # # Assigning results to list of K-fields and K @ Pilot points
    K_PP   = np.ones((n_PP,nreal)) 

    for i in range(nreal):
        K_PP[:,i]       = np.squeeze(result[i][0])


    finish_time = time.perf_counter()
    print("Random field generation finished in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("---")

    # =========================================================================
    #### Ensemble Initialization
    # =========================================================================
    
    start_time      = time.perf_counter()
    
    tstp            = 0
    t_enkf          = 300
    t_tot           = 365
        
    Ysim            = np.zeros((len(obs_cell),nreal))
    Xnew            = np.zeros(((n_c+n_PP),nreal))
    Yobs            = np.ones((len(obs_cell),nreal))

    # Hier noch PP xy in ID's umwandeln
    Ensemble = Ensemble(Xnew, Ysim, obs_cell, xyPP, tstp, [], [], ncores)
    
    result = Parallel(n_jobs=ncores)(delayed(Member)(
            ens_dir[i], np.ma.getdata(result[i][1], subok = True),
            idx_ge) for i in range(nreal)
            )
    
    for i in range(len(result)):
        Ensemble.add_member(result[i])


    Ensemble.update_PP(K_PP)
    Ensemble.update_kmean()
    
    finish_time = time.perf_counter()
    print("Ensemble set up in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("---")
    
    # =========================================================================
    #### EnKF preparation & Initial Conditions for members
    # =========================================================================
    start_time = time.perf_counter()
    
    print(np.mean(Ensemble.members[0].model.ic.strt.get_data()))
    Ensemble.set_transient_forcing(Rch_data[0,1]* f_rch, Qpu_data.iloc[[0]])
    for i in range(7):        
        Ensemble.initial_conditions()
        print(np.mean(np.ma.masked_values(Ensemble.members[0].model.ic.strt.get_data(), 1e+30)))
        
    Ensemble.update_hmean()

    finish_time = time.perf_counter()
    print("Initial conditions finished in {} seconds - using multiprocessing"
          .format(finish_time-start_time)
          )
    print("------")
    
    # =============================================================================
    #### EnKF Data Assimilation
    # =============================================================================

    # Dampening factor for states and parameters
    damp_h          = 0.35
    damp_K          = 0.05
    damp            = np.ones((n_c+n_PP))
    damp[0:n_PP]    = damp_K
    damp[n_PP:]     = damp_h

    # Measurement Error variance [m2]
    eps         = 0.01

    # Defining a stressperiod directory <-- This is hard-coded
    ts          = np.ones(t_tot) 

    # Storing matrices for EnKF algorithm
    Ens_K_mean_mat    = np.ones((t_tot,n_c))
    Ens_h_var_mat     = np.ones((t_tot,n_c))
    Ens_h_mean_mat    = np.ones((t_tot,n_c))
    Ens_h_obs_mat     = np.ones((t_tot,Ensemble.nobs))
    Error_mat         = np.ones((t_tot, 3))
    
    date = datetime.date(2017,1,1)
    
    for i in range(t_tot):
        # Counter for stressperiod and timestep
        print("Starting Time Step {}".format(Ensemble.tstp+1))    
        
        # Setting correct transient forcing
        Rch = Rch_data[i,1]
        Qpu = Qpu_data.iloc[[i]]
        
        # Check how many observations are available
        # I guess we can make this a lot prettier
        
        if Obs_data['Date'].dt.strftime('%Y-%m-%d').str.contains(str(date)).any():
            i1      = Obs_data['Date'].dt.strftime('%Y-%m-%d') == str(date)
            interim = Obs_data.iloc[[i1[i1].index[0]]].to_numpy()
            if sum(j > 5 for j in interim[0][1:]) > 8:
                Assimilate  = True
                Obs_t       = np.zeros((len(interim[0][1:][interim[0][1:] > 0]),2))
                print('Hurray')
                Obs_t[:,1]  =  np.asarray(interim[0][1:][interim[0][1:] > 0])
                Obs_active  =  np.asarray(Obs_data.columns[1:][interim[0][1:] > 0])
                for j in range(len(Obs_active)):
                    # print(int(obs_cell[observation]))
                    Obs_t[j,0] = obs_cell[Obs_active[j]]
            else:
                Assimilate = False
        else:
            Assimilate = False
        
        # ================== BEGIN PREDICTION STEP ============================
        start_time = time.perf_counter()
        
        Ensemble.set_transient_forcing(Rch, Qpu)
        Ensemble.predict() 
        
        if Assimilate == False:
            for j in range(Ensemble.nreal):
                Ensemble.members[j].set_hfield(Ensemble.X[Ensemble.nPP:,j])
                        
        finish_time = time.perf_counter()
        print("Prediction step finished in {} seconds - using multiprocessing"
              .format(finish_time-start_time)
              )
        print("---")
        # ================== END PREDICTION STEP ==============================
        
        
        # ================== BEGIN ANALYSIS STEP ==============================
        if Assimilate == True:
            start_time = time.perf_counter()
            X_prime, Y_prime, Cyy = Ensemble.analysis(Obs_t, eps)

            finish_time = time.perf_counter()
            print("Analysis step finished in {} seconds - using sequential processing"
                  .format(finish_time-start_time)
                  )
            
        # ================== END ANALYSIS STEP ================================
        
        # ================== BEGIN UPDATE STEP ================================
            start_time = time.perf_counter()
        
            # Update the ensemble 
            # HERE WE SHOULD OMIT INDIVIDUAL 1e+30 CELLS
            Ensemble.Kalman_update(damp, X_prime, Y_prime, Cyy, Obs_t)
        
            Ensemble.updateK(cellx, celly, ang, lx, cov_mod)
        
            finish_time = time.perf_counter()
            print("Update step finished in {} seconds - using multiprocessing"
                  .format(finish_time-start_time)
                  )   
            print("---")
            Ensemble.update_tstp()
            print("Time Step {} is finished".format(Ensemble.tstp)) 
            date += datetime.timedelta(days=1)
            print("------")
        else:
            Ensemble.update_tstp()
            print("Time Step {} is finished".format(Ensemble.tstp)) 
            date += datetime.timedelta(days=1)
            print("------")
        # ================== END UPDATE STEP ==================================
        
        # ================== Post-Processing ==================================            

        k = 0
        for key in Ensemble.obsloc.keys():
            Ens_h_obs_mat[i,k] = Ensemble.meanh[0, 0, n_PP+int(Ensemble.obsloc[key])]
            k +=1
        
        Ens_K_mean_mat[i,:]   = Ensemble.meank
        Ens_h_mean_mat[i,:]   = Ensemble.meanh
        Ens_h_var_mat[i,:]    = Ensemble.get_varh()
        
        # Obtain model errors
        # Error_mat[i,:] = Ensemble.ole()

        # if i%10 == 0:
        #     # TODO: Fix DISU Flopy PlotMapView Compatibility
        #     Ensemble.compare()
            

    savemat("Ens_K_mean.mat", mdict={'data':Ens_K_mean_mat})
    savemat("Ens_h_mean.mat", mdict={'data':Ens_h_mean_mat})
    savemat("Ens_h_var.mat", mdict={'data':Ens_h_var_mat})
    savemat("Error.mat", mdict={'data':Error_mat})
    