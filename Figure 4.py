import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch       # counting number of files
import matplotlib.colors as colors

def compartement(f, T2, T1, TE, TR):
    """signal for single compartement"""
    return f*np.exp(-TE/T2)*(1-np.exp(-TR/T1))


def signal(f_k, f_b, f_tub, T2_k, T2_b, T2_tub, T1_k, T1_b, T1_tub, TE, TR):
    """total signal of all three compartements"""
    return compartement(f_k, T2_k, T1_k, TE, TR) + compartement(f_b, T2_b, T1_b, TE, TR) + compartement(f_tub, T2_tub, T1_tub, TE, TR)


def fraction(signal, f, T2, T1, TE, TR):
    """signal fraction of a single compartement, i.e., blood or pre-urine (tubular flow)"""
    return compartement(f, T2, T1, TE, TR)/signal


def f_biexp(signal, f_b, f_tub, T2_b, T2_tub, T1_b, T1_tub, TE, TR):
    """Biexponential signal fraction f (blood and pre-urine combined); f = f1 + f2"""
    return (compartement(f_b, T2_b, T1_b, TE, TR) + compartement(f_tub, T2_tub, T1_tub, TE, TR))/signal


def signal_2_compartement(f, T2_k, T2_b, T1_k, T1_b, TE, TR):
    return compartement(1-f, T2_k, T1_k, TE, TR) + compartement(f, T2_b, T1_b, TE, TR)

def AIC_C(RSS, k):
    return 2*k + 2*np.log(RSS)


def load_parameters_biexp(path, slicewise):
    
    # number of files in folder of given paths
    n_files = len(fnmatch.filter(os.listdir(path), '*.txt'))
    
    if slicewise==True:
        # Initialize arrays for parameters
        D = np.zeros((n_files,4))
        f = np.zeros((n_files,4))
        pD = np.zeros((n_files,4))
    
    else:
        # Initialize arrays for parameters
        D = np.zeros(n_files)
        f = np.zeros(n_files)
        pD = np.zeros(n_files)  
    
    # change work directory 
    os.chdir(path)
    dirs = os.listdir(path)
    
    if slicewise==True:
        # Load parameters
        for i,file in enumerate(dirs):
            biexp_ivim = np.loadtxt(file)
            D[i,:] = biexp_ivim[:,0]
            f[i,:] = biexp_ivim[:,1]
            pD[i,:] = biexp_ivim[:,2]
            
        # reshape to 1D
        D = np.reshape(D, (D.shape[0]*D.shape[1]))
        f = np.reshape(f, (f.shape[0]*f.shape[1]))
        pD = np.reshape(pD, (pD.shape[0]*pD.shape[1]))
        
    else:
        # Load parameters
        for i,file in enumerate(dirs):
            biexp_ivim = np.loadtxt(file)
            D[i] = biexp_ivim[0]
            f[i] = biexp_ivim[1]
            pD[i] = biexp_ivim[2]
    
    return D, f, pD


def load_parameters_bothsides_biexp(path_r, path_l, slicewise):
    
    # number of files in folder of given paths
    n_files = len(fnmatch.filter(os.listdir(path_r), '*.txt'))
        
    # Initialize arrays for parameters
    D_r = np.zeros((n_files,4))
    f_r = np.zeros((n_files,4))
    pD_r = np.zeros((n_files,4))
    D_l = D_r
    f_l = f_r
    pD_l = pD_r
    
    D_r, f_r, pD_r = load_parameters_biexp(path_r, slicewise)
    D_l, f_l, pD_l = load_parameters_biexp(path_l, slicewise)
    
    D = np.append(D_r, D_l, axis=0)
    f = np.append(f_r, f_l, axis=0)
    pD = np.append(pD_r, pD_l, axis=0)
    
    return D, f, pD


# load parameters

# Determine kidney: 'left', 'right' or 'both'
kidney = 'both'

slicewise=True
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'Ergebnisse' folder
results_dir = os.path.join(script_dir, 'Ergebnisse')

path_45_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '45_bi')
path_60_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '60_bi')
path_75_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '75_bi')    
path_90_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '90_bi')
    
path_45_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '45_bi')
path_60_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '60_bi')
path_75_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '75_bi')
path_90_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '90_bi')

# Load measured IVIM parameters
# both kidneys
if kidney=='both':
    D_45, f_45, pD_45 = load_parameters_bothsides_biexp(path_45_r, path_45_l, slicewise)
    D_60, f_60, pD_60 = load_parameters_bothsides_biexp(path_60_r, path_60_l, slicewise)
    D_75, f_75, pD_75 = load_parameters_bothsides_biexp(path_75_r, path_75_l, slicewise)
    D_90, f_90, pD_90 = load_parameters_bothsides_biexp(path_90_r, path_90_l, slicewise)

# right kidney
elif kidney=='right':
    D_45, f_45, pD_45 = load_parameters_biexp(path_45_r, slicewise)
    D_60, f_60, pD_60 = load_parameters_biexp(path_60_r, slicewise)
    D_75, f_75, pD_75 = load_parameters_biexp(path_75_r, slicewise)
    D_90, f_90, pD_90 = load_parameters_biexp(path_90_r, slicewise)
    
# left kidney
elif kidney=='left':
    D_45, f_45, pD_45 = load_parameters_biexp(path_45_l, slicewise)
    D_60, f_60, pD_60 = load_parameters_biexp(path_60_l, slicewise)
    D_75, f_75, pD_75 = load_parameters_biexp(path_75_l, slicewise)
    D_90, f_90, pD_90 = load_parameters_biexp(path_90_l, slicewise)

else:
    print('Enter correct kidney.')
   
f_meas_3 = np.concatenate((f_45, f_60, f_75, f_90), axis=0)[:, np.newaxis, np.newaxis]
f_meas_2 = np.concatenate((f_45, f_60, f_75, f_90), axis=0)[:, np.newaxis]


# Parameters (time in ms)
TE = np.array([45, 60, 75, 90])
repetitions = [f_45.size, f_60.size, f_75.size, f_90.size]
TE = np.repeat(TE, repetitions)
TE_2 = TE[:, np.newaxis]
TE = TE[:, np.newaxis, np.newaxis]
TR = 4500
T1_k, T1_b, T1_tub = 1194, 1600, 4000
T2_k, T2_b, T2_tub = 80, 70, 2000

# create meshgrid of f, f_b and f_tub
f = np.linspace(0, 0.5, 100)
f_b = np.linspace(0, 0.5, 100)
f_tub = np.linspace(0, 0.5, 100)
f_b, f_tub = np.meshgrid(f_b, f_tub)
f_k = 1-f_b-f_tub

# Signal and f of the 3 compartement model
S_3 = signal(f_k, f_b, f_tub, T2_k, T2_b, T2_tub, T1_k, T1_b, T1_tub, TE, TR)
f_model_3 = f_biexp(S_3, f_b, f_tub, T2_b, T2_tub, T1_b, T1_tub, TE, TR)

# Signal and f of the 2 compartement model
S_2 = signal_2_compartement(f, T2_k, T2_b, T1_k, T1_b, TE_2, TR)
f_model_2 = fraction(S_2, f, T2_b, T1_b, TE_2, TR)

# Root mean square (loss function) of the 3 and 2 compartement models
RSS_3 = np.sum((f_model_3-f_meas_3)**2, axis=0)
RSS_2 = np.sum((f_model_2-f_meas_2)**2, axis=0)

# Minimum indices of 3 and 2 compartement models
min_ind = np.unravel_index(RSS_3.argmin(), RSS_3.shape)
min_ind_2 = np.unravel_index(RSS_2.argmin(), RSS_2.shape)
print(f_b[min_ind])
print(f_tub[min_ind])
print(f[min_ind_2])


# Plot
fig = plt.figure(figsize=(14,6))

# 3D plot
ax3d = fig.add_subplot(1,2,1,projection='3d')
surf = ax3d.plot_surface(f_b, f_tub, RSS_3/np.max(RSS_3), cmap='turbo')
# ax3d.tick_params(labelsize=15)
cbar = fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)
cbar.ax.tick_params(labelsize=12)

# Adding labels
ax3d.set_xlabel('blood signal fraction', fontsize=12)
ax3d.set_ylabel('pre-urine signal fraction', fontsize=12)
ax3d.set_zlabel('Normalized RSS', fontsize=12)


# 2D plot
ax2d = fig.add_subplot(1,2,2)
pc = ax2d.pcolormesh(f_b, f_tub, RSS_3/np.max(RSS_3), cmap='turbo', shading='auto',
                    norm=colors.LogNorm())
ax2d.plot(f_b[min_ind], f_tub[min_ind], marker='x', color='white', markersize=10, mew=5)
ax2d.set_xlabel("blood signal fraction", fontsize=15)
ax2d.set_ylabel("pre-urine signal fraction", fontsize=15)
cbar_pc = fig.colorbar(pc, ax=ax2d)
cbar_pc.ax.tick_params(labelsize=12)

ax2d.tick_params(labelsize=12)


# Get minimum values
# Get the indices of the minimum values along axis=0
min_indices = np.argmin(RSS_3, axis=0)

# Get the corresponding f_tub values for the minimum RSS values
min_f_tub_values = f_tub[min_indices, 0]

# Get minimum RSS values
min_RSS_3 = np.min(RSS_3/np.max(RSS_3), axis=0)

# Overlay the points where the minima occur
# plt.plot(f_b[0,:50], min_f_tub_values[:50], color='red')

# fig = plt.figure()
# plt.plot(f_b[0,:50], min_RSS_3[:50])
# plt.tick_params(labelsize=20)
    
# fig, ax = plt.subplots()
# ax.plot(section_z/np.max(RSS_3))

# fig = plt.figure()

# plt.plot(TE[:,0,0], f_meas[:,0,0], linestyle='None', marker='o', label='measurement')
# plt.plot(TE[:,0,0], f_model[:, min_ind[0], min_ind[1]], label='3 compartement model')
# plt.xlabel('TE (ms)', fontsize=20)
# plt.ylabel('f', fontsize=20)
# plt.tick_params(labelsize=20)

# plt.legend(fontsize=20)

plt.tight_layout()
plt.show()