#! /usr/bin/env ipython
##############################################################
##                                                          ##
##  Generate Abernathy Style Forcing, by helen burns        ##
##                                                          ##
##############################################################
from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
import math
from pylab import *
from IPython.display import display, Math, Latex
import glob


# When writing in python it is very important to note reverse dimensions!!
# MITgcm assumes column major order (as does matlab) Python, uses row major order.
# Mosty it's fine just to write straight to binary, but to absolutely specific of the format for MITgcm the WriteFile fuction (taken from the MITgcm csv gendata.py):

# Use writeFile to write files in the correct format!
# Alessandro I have made it so you can use this function
sys.path.append('/noc/users/hb1g13/Python/python_functions/')
from Writebin import *


### Decide parameters:
'''Resolution 
Depth
Domain
Boundary Condition
Topography
Forcing'''
                
Topo="Flat" #Please Choose ridge, slope or flat
Wind="Standard" # Sine bell 0.2N/m$^2$
Heat="Abenathey" # Please Choose Abernathey or nonetQ
BC="Sponge" # Please Choose Sponge or Diffusion
#Name="Sponge" # Give Experiment Name

# Adjust accordingly
Res=20000
Ly=2000e3
Lx=1000e3 #Full domain = 4000km otherwise 1000km
H=2985 # Diffusion = 3800m, Sponge = 2985m
nz=30 # Diffusion = 24 level, Sponge= 30 levels


#x="/noc/users/hb1g13/MITgcm/Mobilis/"+Name+"/input/" 
#os.chdir(x)


### Set up grid:

#Dimensions
nx=np.round(Lx/Res)
ny=np.round(Ly/Res)
dx=np.ones(nx)*Res
dy=np.ones(ny)*Res
#Write binary output
writeFile('delY',dy)
writeFile('delX',dx)
# Create c-grid with grid points in dead center
x=(np.cumsum(dx)-dx/2)-Lx/2
y=(np.cumsum(dy)-dy/2)-Ly/2
[Y, X]=np.meshgrid(y,x) 


### Now Create topography:

# Start with flat, then add slope and ridges

h= -H*np.ones((nx,ny)) # Flat bottom
if Topo=="ridge":#2500 and 2000 for full depth
         h= h+(1500 + 300*np.sin(10*pi*Y/Ly) + 400*np.sin(8*pi*Y/Ly)+ 300*sin(25*pi*Y/Ly) )*(1/np.cosh(((X)-0.2*Y+3e5)/1.2e5))
         h= h+((1000 + 600*np.sin(11*pi*Y/Ly) + 300*np.sin(7*pi*Y/Ly)+ 500*sin(21*pi*Y/Ly) )*(1/np.cosh(((X)+0.1*Y+1.5e6)/1.2e5)))
if Topo=="slope" or Topo=="ridge":
    for i in range(int(nx)):
      slope= np.transpose(H*(np.divide((Y[i,0:round(0.2*ny)]-Y[i,0]),(Y[i,0]-Y[i,round(0.2*ny)]))))
      h2=h[:,0:round(0.2*ny)]
      h[:,0:round(0.2*ny)]=np.maximum(slope,h2)
# Close both ends
h[:,0]=0
h[:,-1]=0
# Write to binary
writeFile('topog',np.transpose(h))

if Topo=="flat" or Topo=="slope":
    plt.plot(y/1000,h[nx/2,:])
    plt.title('Topography')
    plt.ylabel('Depth (m)')
    plt.xlabel('Y (km)')
if Topo=='ridge':
    plt.contourf(x/1000,y/1000,np.transpose(h),30)
    cb=plt.colorbar()
    plt.title('Topography')
    plt.ylabel('Y (km)')
    plt.xlabel('X (km)')
    cb.set_label('Depth (m)')
    
### Surface Heat Forcing

# Now for the surface heat forcing:
# Must have bouyancy gain in the south and bouyancy loss over maximum wind sress to allow overturning

#MITgcm reads in the file as cooling = positive
Q=10*(np.sin(Y*(3*pi/Ly)))
Q[:,ny-(np.round(ny/6)):ny]=0
if Heat=="nonetQ":
   Q=np.zeros(np.shape(Q)) 
   for i in range(0,int(ny/6)):
       Q[:,i]=-10*(np.sin(Y[:,i]*(2*3*pi/Ly)))
   for i in range(int(ny/6),int(4*ny/6)): 
       Q[:,i]=10*(np.sin(Y[:,i]*(3*pi/Ly)))
   for i in range(int(3*ny/6),int(4*ny/6)): 
       Q[:,i]=10*(np.sin(Y[:,i]*(2*3*pi/Ly)))
   Q=Q+(-sum(Q)/(ny*nx)) 
# Write to binary
writeFile('Qsurface',np.transpose(Q))
# netcdf check
f=netcdf.netcdf_file('Qsurface.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
Q2=f.createVariable('Q','float',('X','Y'))
Q2[:]=Q
f.close()

plt.plot(y/1000,Q[10,:])
plt.title('Surface Heat Flux $W/m^2$')
plt.ylabel('Heat Flux ($W/m^2$)')
plt.xlabel('Meridional Distance (m)')


### Windstress

# Plus the Windress with $\tau_o$ set to $0.2Nm^-2$
# 
tau=0.2*((np.sin((Y+Ly/2)*(pi/Ly)))) #Y is centred at 0 so put that back!
if BC=='Diffusion':
    Taunew = tau + 1e-3 * (np.random.random((nx,ny)) - 0.5)
    tau=Taunew
# Write to binary
writeFile('Wind',np.transpose(tau))
# netcdf check
f=netcdf.netcdf_file('Wind.nc','w')
f.createDimension('Xp1',nx+1)
f.createDimension('Y',ny)
tau3=np.zeros((ny,nx+1))
tau3[:,1:]=np.transpose(tau)
tau2=f.createVariable('tau','double',('Xp1','Y'))
tau2[:]=np.transpose(tau3)
f.close()

plt.plot(y/1000,tau[10,:])
plt.title('Surface Wind Stress $N/m^2$')
plt.ylabel('$\tau$ ($N/m^2$)')
plt.xlabel('Meridional Distance (m)')


### Generate Sponge

#                Now creat a Sponge mask and a reference profile to relax to:
                
#Parameters
N=1e3 # Natural stratification
deltaT=8
Tref=np.zeros(nz)
#Create depth array:
a=5,22.5,60
b=np.linspace(135,2535,25)
c=2685,2885
z=np.concatenate([a,b,c])

Tref = deltaT*(exp(-z/N)-exp(-H/N))/(1-exp(-H/N))

plt.plot(Tref,z)
plt.gca().invert_yaxis()
plt.title('Temperature Profile')
plt.ylabel('Depth (m)')
plt.xlabel('Temperature $^oC$')

#Make a 3D array of it
T=np.ones((nz,ny,nx))
Temp_field=np.zeros(np.shape(T))
for i in range(int(nx)):
    for j in range(int(ny)):
        Temp_field[:,j,i]=np.multiply(Tref,T[:,j,i])

Tnew = transpose(tile(Temp_field.mean(axis=2),(nx,1,1)),[1,2,0])
Tnew[:,-1] = Tnew[:,-2]
#Maybe add more 
if BC=='Diffusion':
    Tnew = Tnew + 1e-3 * (np.random.random((nz,ny,nx)) - 0.5)
else:
    Tnew = Tnew + 1e-3 * (np.random.random((nz,ny,nx)) - 0.5)

# Write to binary
writeFile('T_Sponge',Temp_field)
writeFile('T.init',Tnew)
# netcdf check
f=netcdf.netcdf_file('TSponge.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
f.createDimension('Z',nz)
Temp=f.createVariable('Temp','double',('Z','Y','X'))
Temp[:]=Temp_field
f.close()

#Make 3D mask
#Must vary between 0 (no Relaxation) and 1 (full relaxtion)
#I have gone for a parabolic decay in x and linear decay in z (from playing around)
msk=np.zeros(np.shape(T))
for k in range(0,len(z)):
    for i in range(len(x)):  
        msk[k,ny-int(ny/20):ny,i]=((np.divide((Y[i,ny-int(ny/20)-1:ny-1]-Y[i,int(ny/20)-1]),(Y[i,ny-1]-Y[i,int(ny/20)-1])))) 
# Write to binary
writeFile('T.msk',msk)
# netcdf check
f=netcdf.netcdf_file('Mask.nc','w')
f.createDimension('X',nx)
f.createDimension('Y',ny)
f.createDimension('Z',nz)
Mask=f.createVariable('Mask','double',('Z','Y','X'))
Mask[:]=(msk)
f.close()

plt.contourf(y/1000,z,msk[:,:,10],24,cm=cm.Spectral)
cbar = plt.colorbar()
plt.gca().invert_yaxis()
plt.title('Mask Matrix')
plt.ylabel('Depth (m)')
plt.xlabel('Meridional Distance (km)')

if BC=="Diffusion":
        #Background
        diffusi=(1e-5)*np.ones((nz,ny,nx))
        # Linear ramp
        for k in range(0,nz):
           for i in range(0,int(nx)):
               diffusi[k,ny-20:ny,i]=0.0005+500*(np.divide((Y[i,ny-21:ny-1]-Y[i,ny-21]),\
                                                (Y[i,ny-1]-Y[i,ny-21])))*diffusi[k,ny-21:ny-1,i]
               # Enhance at the surface
        for k in range(0,3):
            for i in range(0,int(nx)):
                diffusi[k,:,i]=np.maximum(0.002*((z[nz-1-k]/H)**2)                                          *(1-np.divide(2*abs(Y[i,:]),(2*Ly))),diffusi[k,:,i])
        # Write to binary
        writeFile('diffusi.bin',diffusi)
        # netcdf check
        f=netcdf.netcdf_file('diffusi.nc','w')
        f.createDimension('Z',nz)
        f.createDimension('Y',ny)
        f.createDimension('X',nx)
        Diff=f.createVariable('Diffusi','double',('Z','Y','X'))
        Diff[:]=diffusi
        f.close()
        plt.contourf(y/1000,z,diffusi[:,:,150],24,cm=cm.Spectral)
        cbar = plt.colorbar()
        plt.gca().invert_yaxis()
        

