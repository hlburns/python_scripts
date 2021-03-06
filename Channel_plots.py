# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:11:46 2016
Channel Plots

@author: hb1g13
"""

# Some parameters
VAR = 'Psi'  # Pick what plot
Full = 'N'  # 9 Pannels isn't ideal for presentations N option give 4 plots
Year = 'PSI.nc'
Qplot = 'Y'
# Load in Modules
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.ma as ma
import sys
from matplotlib import gridspec
sys.path.append('/noc/users/hb1g13/Python/python_functions/')
from HB_Plot import nf, fmt
import SG
import useful as hb
import layers_calc_numba
sys.path.append('/noc/users/hb1g13/Python/python_functions/MITgcmUtils/')
import utils

if Full == 'N':
    tau = ['3', '300', '3000', 'Closed']
elif Full == 'Extremes':
    tau = ['3','Closed']
else:
    tau = ['3', '10', '30', '100', '300',
           '1000', '3000', '10000', 'Closed']
Figletter = ['a) ','b) ','c) ','d) ','e)','f)','g)','h)','j)']

# Path root
x = '/noc/msm/scratch/students/hb1g13/Mobilis'
# Now Make file structure
check = 0
runs = []
for i in range(len(tau)):
    flist = x+'/'+str(tau[i])+'daynokpp/VSQ.nc'
    if not os.path.exists(flist):
        print ' WARNING: '+flist+' does not exist! (skipping this tau...)'
        check += 0
    else:
        check += 1
        runs.append(i)
Runs=np.array(runs)

# Set up some common feautres
fname = x+'/3daynokpp/'
c = utils.ChannelSetup(output_dir=str(fname))

#Reference values
tRef = [7.95797596, 7.81253554, 7.50931741, 6.93595077, 6.23538398,
        5.60148493, 5.02790935, 4.5089167 , 4.03931274, 3.6143975 ,
        3.22991829, 2.88202712, 2.56724216, 2.28241296, 2.02468884,
        1.79149041, 1.58048375, 1.38955702, 1.21679938, 1.0604818 ,
        0.9190398 , 0.79105779, 0.67525488, 0.57047207, 0.47566066,
        0.38987176, 0.31224674, 0.24200873, 0.14898305, 0.04478574]
        
#Suface Heat Flux
Q2_levs = (np.arange(-1,8,1))
Q = c.mnc('SURF.nc', 'oceQnet').mean(axis=0).mean(axis=1)

# Figure arrangement
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 4, 4])

fig = plt.figure(figsize=(16.5, 16.5))

eke_levs = np.arange(-0., .15, .015)
eke_ticks =np.arange(-0., .15, .03)

def add_SHF():
    """ Add surface heat flux to top of figure  """
    for i in range(2):
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 4, 4])
        ax = plt.subplot(gs[i])
        ax.plot(c.yc/1000,-Q, color='k', linewidth=2)
        ax.set_title('Surface Heat Flux $W/m^2$', fontsize=20)
        ax.set_ylabel('Heat Flux ($W/m^2$)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

EKEt = []
for i in range(len(Runs)):
    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    # PLOT PANNELS
    EKE = c.calc_EKE()
    EKEt.append(np.sum(EKE))
    ax = plt.subplot(gs[2+i])
    p = ax.contourf(c.yc/1000, c.zc/1000,
                EKE, eke_levs, cmap=plt.cm.plasma_r, extend='both')
    # Add iso therms
    Tavlat = c.get_zonal_avg('Tav.nc','THETA')
    if str(tau[Runs[i]]) == 'Closed':
        Tavlat = np.apply_along_axis(np.divide, 0, Tavlat, 4-np.sum(tRef*c.dzf)/-c.zc[-1])
    q = ax.contour(c.yc/1000, c.zc/1000, Tavlat, Q2_levs, colors='k', linewidths=2)
    q.levels = [nf(val) for val in q.levels]
    plt.clabel(q, q.levels[::2], inline=1, fmt=fmt, fontsize=25)
    # Add diabatic layer depth
    PI = c.mnc('PSI.nc',"LaPs1TH").mean(axis=2)
    PI = ma.masked_array(PI,PI<0.95)
    # Depths
    th = c.mnc('PSI.nc',"LaHs1TH").mean(axis=2)
    depths = np.cumsum(th[::-1],axis=0)[::-1]
    DDL = np.zeros(len(c.yc))
    psi = c.get_psi_iso()
    for jj in range(len(c.yc)):
        if ma.all(PI[:,jj]  == 1)  or np.all(psi[:,jj] == -0) or PI[:,jj].mask.all():
            continue
        indx = ma.nonzero(PI[:,jj]<0.9999999999)[0]
        a = indx[np.nonzero(indx>3)[0]][0]
        if a<41 and depths[a-1,jj] - depths[a,jj] > 150:
            DDL[jj] = (depths[a-1,jj]+depths[a,jj])/2
        else:
            DDL[jj] = depths[a,jj]
   
    r = ax.plot(c.yc/1000,SG.savitzky_golay(-DDL/1000,21,1),
                color='0.75', linewidth=4)
    # Lables
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+'day', fontsize=30)
    if str(tau[Runs[i]]) == 'Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]), fontsize=30)
    ax.set_xlabel('Distance (km)', fontsize=30)
    ax.set_ylabel('Depth (km)', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    
plt.tight_layout()
cax = fig.add_axes([1, 0.1, 0.03, 0.8])
cbar = fig.colorbar(p, cax=cax, ticks=eke_ticks)
cbar.ax.set_ylabel('EKE J/kg', fontsize=30, )
cbar.ax.tick_params(labelsize=30)
plt.savefig('/noc/users/hb1g13/Figures/EKE.png')

# MOC 
Psi_levs = np.arange(0.05, 2.8, .3)
Psi_ticks =np.arange(0, 3., .5)
for i in range(len(Runs)):
    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    psi = c.get_psi_bar()/10**6
    Tavlat = c.mnc('Tav.nc','THETA').mean(axis=2)
    y = c.yc/1000
    yp = c.yg/1000
    ax = fig.add_subplot(2, 2, i+1)
    p = ax.contourf(yp, c.zc/1000, psi, Psi_levs,
                    cmap=plt.cm.Reds,extend='both')
    q = ax.contour(y, c.zc/1000,
                   Tavlat, Q2_levs, colors='k', linewidths=2)
    q.levels = [nf(val) for val in q.levels]
    plt.clabel(q, q.levels[::2], inline=1, fmt=fmt, fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+' day', fontsize=30)
    if str(tau[Runs[i]]) == 'Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]), fontsize=30)
    ax.set_xlabel('Distance (km)', fontsize=30)
    ax.set_ylabel('Depth (km)', fontsize=30)
plt.tight_layout()
cbar = fig.colorbar(p, cax=cax, ticks=Psi_ticks)
cbar.ax.set_ylabel('$\Psi$ (Sv)', fontsize=30)
cbar.ax.tick_params(labelsize=30)
plt.savefig('/noc/users/hb1g13/Figures/MOC.png')

# RMOCT
for i in range(len(Runs)):
    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    g = layers_calc_numba.LayersComputer(c)
    Rho = g.glvl
    psi = c.get_psi_iso()/10**6
    y = c.yg/1000
    ax = fig.add_subplot(2, 2, i+1)
    p = ax.contourf(y,Rho,psi,Psi_levs,cmap=plt.cm.seismic,extend='both') 
    plt.ylim(-0.2,10.4)
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+'day',fontsize=40)
    if str(tau[Runs[i]])=='Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]),fontsize=40)
    ax.set_xlabel('Distance (km)',fontsize=40)
    ax.set_ylabel('Temperature $^o$C',fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=40)
plt.tight_layout()
# Add Colorbar
cbar = fig.colorbar(p, cax=cax,ticks=Psi_ticks)
cbar.ax.set_ylabel('$\Psi \,\, (sv)$',fontsize=40)
cbar.ax.tick_params(labelsize=40)
plt.savefig('/noc/users/hb1g13/Figures/RMOCT.png')

#RMOC (y,Z)
for i in range(len(Runs)):
    fname = x+'/'+str(tau[Runs[i]])+'daynokpp/'
    c = utils.ChannelSetup(output_dir=str(fname))
    # PLOT PANNELS
    Psi = c.get_psi_iso_z()/10**6
    ax = plt.subplot(gs[2+i])
    p = ax.contourf(c.yc/1000, c.zc/1000,
                Psi, Psi_levs, cmap=plt.cm.seismic, extend='both')
    Tavlat = c.get_zonal_avg('Tav.nc','THETA')
    if str(tau[Runs[i]]) == 'Closed':
        Tavlat = np.apply_along_axis(np.divide, 0, Tavlat, 4-np.sum(tRef*c.dzf)/-c.zc[-1])
    q = ax.contour(c.yc/1000, c.zc/1000, Tavlat, Q2_levs, colors='k', linewidths=2)
    q.levels = [nf(val) for val in q.levels]
    plt.clabel(q, q.levels[::2], inline=1, fmt=fmt, fontsize=25)
     # Layer probability mask 
    g = layers_calc_numba.LayersComputer(c)
    PI = c.mnc('PSI.nc',"LaPs1TH").mean(axis=2)
    PI = ma.masked_array(PI,PI<0.95)
    th = c.mnc('PSI.nc',"LaHs1TH").mean(axis=2)
    depths = np.cumsum(th[::-1],axis=0)[::-1]
    
    # Find Max ROC and depth of diabatic layer
    DDL = np.zeros(len(c.yc))
    DDL_matrix = np.zeros_like(Tavlat)
    for jj in range(len(c.yc)):
        if ma.all(PI[:,jj]  == 1)  or np.all(Psi[:,jj] == -0) or PI[:,jj].mask.all():
            continue
        indx = ma.nonzero(PI[:,jj]<0.9999999999)[0]
        b = indx[np.nonzero(indx>3)[0]]
        if len(b)>=2 and (b[1]-b[0])>1:
            a = b[0]
        else:
            a = b[0]
            
        if a<41 and depths[a-1,jj] - depths[a,jj] > 150:
            DDL[jj] = (depths[a-1,jj]+depths[a,jj])/2
        else:
            DDL[jj] = depths[a,jj]
   
    ax.axis('tight')
    r = ax.plot(c.yc/1000,SG.savitzky_golay(-DDL/1000,21,1),
                color='0.65', linewidth=4)
    ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]])+'day', fontsize=30)
    if str(tau[Runs[i]]) == 'Closed':
        ax.set_title(str(Figletter[Runs[i]])+str(tau[Runs[i]]), fontsize=30)
    ax.set_xlabel('Distance (km)', fontsize=30)
    ax.set_ylabel('Depth (km)', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
plt.tight_layout()
cbar = fig.colorbar(p, cax=cax, ticks=Psi_ticks)
cbar.ax.set_ylabel('$\Psi$  (Sv)', fontsize=30, )
cbar.ax.tick_params(labelsize=30)
plt.savefig('/noc/users/hb1g13/Figures/ReMapped.png')
