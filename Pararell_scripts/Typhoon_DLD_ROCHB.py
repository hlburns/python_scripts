# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:53:45 2016

@author: hb1g13
"""

from IPython import parallel
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import netCDF4
import numpy.ma as ma
from pylab import *
sys.path.append('/noc/users/hb1g13/Python/python_functions/')
import SG as SG
import layers_calc_numba
sys.path.append('/noc/users/hb1g13/Python/python_functions/MITgcmUtils/')
import utils

rc = parallel.Client('/noc/users/hb1g13/.ipython/profile_typhoon/security/ipcontroller-client.json')
dv = rc[:]
rc.ids

dv.execute('import sys')
dv.execute('sys.path.append("/noc/users/hb1g13/Python/python_functions/")')
dv.execute('import layers_calc_numba')
dv.execute('sys.path.append("/noc/users/hb1g13/Python/python_functions/MITgcmUtils/")')
dv.execute('import utils')
# Some parameters
Full = 'N'  # 9 Pannels isn't ideal for presentations N option give 4 plots
if Full == 'N':
    tau = ['3', '300', '3000', 'Closed']
elif Full == 'Extremes':
    tau = ['3','Closed']
else:
    tau = ['3', '10', '30', '100', '300',
           '1000', '3000', '10000', 'Closed']
Figletter = ['a) ','b) ','c) ','d) ','e)','f)','g)','h)','j)']
# Path root
dv.execute("x = '/noc/msm/scratch/students/hb1g13/Mobilis'")
x = '/noc/msm/scratch/students/hb1g13/Mobilis'
# Now Make file structure
check = 0
runs = []
for i in range(len(tau)):
    flist = x+'/'+str(tau[i])+'daynokpp/PSI.nc'
    if not os.path.exists(flist):
        print ' WARNING: '+flist+' does not exist! (skipping this tau...)'
        check += 0
    else:
        check += 1
        runs.append(i)
Runs=np.array(runs)

fig = plt.figure(figsize=(18.5, 16.5))
for i in range(len(Runs)):

    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    g = layers_calc_numba.LayersComputer(c)
    # ROC 
    psi = c.get_psi_iso()
    # Layer probability mask 
    PI = c.mnc('PSI.nc',"LaPs1TH").mean(axis=2)
    PI = ma.masked_array(PI,PI<0.95)
    #psi = ma.masked_array(psi, PI < .98 )
    # Depths
    th = c.mnc('PSI.nc',"LaHs1TH").mean(axis=2)
    depths = np.cumsum(th[::-1],axis=0)[::-1]
    
    # Find Max ROC and depth of diabatic layer
    DDL = np.zeros(len(c.yc))

    for jj in range(len(c.yc)):
        if ma.all(PI[:,jj]  == 1)  or np.all(psi[:,jj] == -0) or PI[:,jj].mask.all():
            continue
        indx = ma.nonzero(PI[:,jj]<1)[0]
        b = indx[np.nonzero(indx>3)[0]]
        if len(b)>=2 and (b[1]-b[0])>1:
            a = b[1]
        else:
            a = b[0]
        if a<41 and depths[a-1,jj] - depths[a,jj] > 150:
            a = a-1
        DDL[jj] = depths[a,jj]

    ax = fig.add_subplot(2, 2, i+1)
    p = plt.plot(c.yc/1000, SG.savitzky_golay(-DDL,31,1), 'r', linewidth=3)
    plt.ylim(-2895,0)
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+'day', fontsize=30)
    if str(tau[Runs[i]]) == 'Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]), fontsize=30)
    ax.set_xlabel(r'$km$', fontsize=30)
    ax.set_ylabel(r'Depth', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)

plt.tight_layout()
plt.savefig(x+'/Figures/Diabatic_layer_depth.png')

fig = plt.figure(figsize=(18.5, 16.5))
for i in range(len(Runs)):

    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    g = layers_calc_numba.LayersComputer(c)
    # ROC 
    psi = c.get_psi_iso()
    # Layer probability mask 
    PI = c.mnc('PSI.nc',"LaPs1TH").mean(axis=2)
    PI = ma.masked_array(PI,PI<0.95)
    #psi = ma.masked_array(psi, PI < .98 )
    # Depths
    th = c.mnc('PSI.nc',"LaHs1TH").mean(axis=2)
    depths = np.cumsum(th[::-1],axis=0)[::-1]
    
    # Find Max ROC and depth of diabatic layer
    DDL = np.zeros(len(c.yc)) 
    ROC = np.zeros(len(c.yc)) 
    for jj in range(len(c.yc)):
        if ma.all(PI[:,jj]  == 1)  or np.all(psi[:,jj] == -0) or PI[:,jj].mask.all():
            continue
        indx = ma.nonzero(PI[:,jj]<1)[0]
        b = indx[np.nonzero(indx>3)[0]]
        if len(b)>=2 and (b[1]-b[0])>1:
            a = b[1]
        else:
            a = b[0]
        if a<41 and depths[a-1,jj] - depths[a,jj] > 150:
            a = a-1
        DDL[jj] = depths[a,jj]
        if psi[a,jj]/10**6 > 1.0:
            a = a-1
        ROC[jj] = psi[a,jj]
    Q = c.mnc('SURF.nc', 'oceQnet').mean(axis=0).mean(axis=1)
    Psipred = (Q)/(1000 * 3985)
    ax = fig.add_subplot(2, 2, i+1)
    p = plt.plot(c.yc/1000, SG.savitzky_golay(ROC,41,1)/10**6, 'k', linewidth=2)
    q = plt.plot(c.yc/1000, (Psipred/c.dzc[0])*10**6, 'r', linewidth=3)
    plt.ylim(-.8,.8)
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+'day', fontsize=30)
    if str(tau[Runs[i]]) == 'Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]), fontsize=30)
    ax.set_xlabel(r'$km$', fontsize=30)
    ax.set_ylabel(r'Depth', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)

plt.tight_layout()
plt.savefig(x+'/Figures/ROC_at_DLD.png')

fig = plt.figure(figsize=(18.5, 16.5))

for i in range(len(Runs)):
    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    g = layers_calc_numba.LayersComputer(c)
    # Calculate cartesian diabatic eddies 
     # PLOT PANNELS
    CellVol = c.rac * np.tile(c.dzf, (c.Nx, c.Ny, 1)).T
    dv['fname'] = fname
   
    # load V, W, T bar
    # put everything on the C Grid
    VT = (c.mnc('Tav_VT.nc', 'VVELTH'))
    WT = (c.mnc('Tav_VT.nc', 'WVELTH'))
    Tv = utils.cgrid_to_vgrid(c.mnc('Tav.nc', 'THETA'))
    Tw = utils.cgrid_to_wgrid(c.mnc('Tav.nc', 'THETA'))
    V = (c.mnc('Tav.nc', 'VVEL'))
    W = (c.mnc('Tav.nc', 'WVEL'))
    T = c.mnc('Tav.nc', 'THETA')
    npad = ((0, 1), (0, 0), (0, 0))
    W = np.pad(W, pad_width=npad, mode='constant', constant_values=0)
    WT = np.pad(WT, pad_width=npad, mode='constant', constant_values=0)
    VTbar = V * Tv
    WTbar = W * Tw
    VpTp = VT - VTbar
    WpTp = WT - WTbar
    
    # Vertical Mass-Weight Transp of Pot Temp (K.m/s)
    WTHMASS = c.mnc('SURF.nc', 'WTHMASS')
    # Surface cor
    Surcor = (WTHMASS.mean(axis=0)).mean(axis=1)
    Ty = c.ddy_cgrid_centered(T)
    Tz = c.ddz_cgrid_centered(T)

    Sp = - Ty/Tz
    dv['DEs'] = c.wgrid_to_cgrid(WpTp) - c.vgrid_to_cgrid(VpTp)*Sp
    #T = c.mnc('Tav.nc', 'THETA').mean(axis=2)
    dv.execute('c = utils.ChannelSetup(output_dir=str(fname))',block='True')
    dv.execute('g = layers_calc_numba.LayersComputer(c)',block='True')
    dv.execute('A_local=g.interp_to_g(DEs,c.mnc("Tav.nc", "THETA"))',block='True')[0]
    DE_l = dv.gather('A_local').get()[0]
    DE_l = DE_l.mean(axis=2)
    # Calculate diabatic layer depth and ROC 
    f = netCDF4.Dataset(str(fname)+'DEyz2yT.nc','w')
    f.createDimension('T',len(g.glvl))
    f.createDimension('Y',c.Ny)
    h2=f.createVariable('DE_l','float',('Y','T'))
    h2[:] = DE_l
    h2.standard_name = 'WpTp -VpTp*Sp remamped to T space'
    f.close()
    # ROC 
    psi = c.get_psi_iso()
    # Layer probability mask 
    PI = c.mnc('PSI.nc',"LaPs1TH").mean(axis=2)
    PI = ma.masked_array(PI,PI<0.95)
    #psi = ma.masked_array(psi, PI < .98 )
    # Depths
    th = c.mnc('PSI.nc',"LaHs1TH").mean(axis=2)
    depths = np.cumsum(th[::-1],axis=0)[::-1]
    

    # Find Max ROC and depth of diabatic layer
    DDL = np.zeros(len(c.yc)) 
    ROC = np.zeros(len(c.yc)) 
    TL = np.zeros(len(c.yc))
    DE_dl = np.zeros(len(c.yc)) 
    for jj in range(len(c.yc)):
        if ma.all(PI[:,jj]  == 1)  or np.all(psi[:,jj] == -0) or PI[:,jj].mask.all():
            continue
        indx = ma.nonzero(PI[:,jj]<1)[0]
        b = indx[np.nonzero(indx>3)[0]]
        if len(b)>=2 and (b[1]-b[0])>1:
            a = b[1]
        else:
            a = b[0]
        if a<41 and depths[a-1,jj] - depths[a,jj] > 150:
            a = a-1
        DDL[jj] = depths[a,jj]
        DE_dl[jj] = DE_l[a,jj]
        TL[jj] = g.glvl[a-1]
        if psi[a,jj]/10**6 > 1.0:
            a = a-1
        ROC[jj] = psi[a,jj]
    Temp = c.mnc('Tav.nc','THETA').mean(axis=2)
    TLZ =(np.tile((TL+Temp[1,:])/2,(c.Nz,1)))
    TLy = c.ddy_cgrid_centered(TLZ)[0,:]
    Q = c.mnc('SURF.nc', 'oceQnet').mean(axis=0).mean(axis=1)
    Psipred = (Q)/(1000 * 3985)
    ax = fig.add_subplot(2, 2, i+1)
    p = plt.plot(c.yc/1000, SG.savitzky_golay(ROC,41,1), 'k', linewidth=2)
    q = plt.plot(c.yc/1000, (Psipred/c.dzc[0])*10**6, 'r', linewidth=3)
    #r = plt.plot(c.yc/1000, (SG.savitzky_golay(ROC,41,1)*SG.savitzky_golay(Ty,41,1)-(Psipred/c.dzc[0])*10**6), 'b', linewidth=1)
    q = plt.plot(c.yc/1000, -SG.savitzky_golay(DE_dl,21,1)*10**6,'b',linewidth=3)
    plt.ylim(-1.5,1.5)
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+'day', fontsize=30)
    if str(tau[Runs[i]]) == 'Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]), fontsize=30)
    ax.set_xlabel(r'$km$', fontsize=30)
    ax.set_ylabel(r'DEs_remaped_to_L', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)

plt.tight_layout()
plt.savefig(x+'/Figures/PHB_1.png')