import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch       # counting number of files
from scipy import stats   # Statistical analysis


def load_parameters_biexp(path, slicewise):
    
    # number of files in folder of given paths
    n_files = len(fnmatch.filter(os.listdir(path), '*.txt'))
    
    if slicewise==True:
        # Initialize arrays for parameters
        D = np.zeros((n_files,4))
        f = np.zeros((n_files,4))
        pD = np.zeros((n_files,4))
        # D = np.zeros((n_files,2))
        # f = np.zeros((n_files,2))
        # pD = np.zeros((n_files,2)
    
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
    
    return D*1000, f*100, pD*1000


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

# determine slicewise or volunteer-wise
slicewise=True
# Determine kidney: 'left', 'right' or 'both'
kidney = 'both'

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'Ergebnisse' folder
results_dir = os.path.join(script_dir, 'Ergebnisse')

# Paths where fitted IVIM parameters are saved
path_45_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '45_bi')
path_60_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '60_bi')
path_75_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '75_bi')
path_90_r = os.path.join(results_dir, 'Cortex + Medulla', 'rechts', '90_bi')
    
path_45_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '45_bi')
path_60_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '60_bi')
path_75_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '75_bi')
path_90_l = os.path.join(results_dir, 'Cortex + Medulla', 'links', '90_bi')


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
    
    
# Print mean and std
print("\nMean +- std")

# 45 ms
print("45 ms: ")
print("D =", np.mean(D_45),"+-", np.std(D_45))
print("D* =", np.mean(pD_45),"+-", np.std(pD_45))
print("f =", np.mean(f_45),"+-", np.std(f_45))

# 60 ms
print("60 ms: ")
print("D =", np.mean(D_60),"+-", np.std(D_60))
print("D* =", np.mean(pD_60),"+-", np.std(pD_60))
print("f =", np.mean(f_60),"+-", np.std(f_60))

# 75 ms
print("75 ms: ")
print("D =", np.mean(D_75),"+-", np.std(D_75))
print("D* =", np.mean(pD_75),"+-", np.std(pD_75))
print("f =", np.mean(f_75),"+-", np.std(f_75))

# 90 ms
print("90 ms: ")
print("D =", np.mean(D_90),"+-", np.std(D_90))
print("D* =", np.mean(pD_90),"+-", np.std(pD_90))
print("f =", np.mean(f_90),"+-", np.std(f_90))


# Print median, Q1 and Q3
print("\nMedian [Q1, Q3]")

# 45 ms
print("45 ms: ")
print("D =", np.median(D_45),"[", np.quantile(D_45, 0.25),",", np.quantile(D_45, 0.75), "]")
print("f =", np.median(f_45),"[", np.quantile(f_45, 0.25),",", np.quantile(f_45, 0.75), "]")
print("D* =", np.median(pD_45),"[", np.quantile(pD_45, 0.25),",", np.quantile(pD_45, 0.75), "]")

# 60 ms
print("60 ms: ")
print("D =", np.median(D_60),"[", np.quantile(D_60, 0.25),",", np.quantile(D_60, 0.75), "]")
print("f =", np.median(f_60),"[", np.quantile(f_60, 0.25),",", np.quantile(f_60, 0.75), "]")
print("D* =", np.median(pD_60),"[", np.quantile(pD_60, 0.25),",", np.quantile(pD_60, 0.75), "]")

# 75 ms
print("75 ms: ")
print("D =", np.median(D_75),"[", np.quantile(D_75, 0.25),",", np.quantile(D_75, 0.75), "]")
print("f =", np.median(f_75),"[", np.quantile(f_75, 0.25),",", np.quantile(f_75, 0.75), "]")
print("D* =", np.median(pD_75),"[", np.quantile(pD_75, 0.25),",", np.quantile(pD_75, 0.75), "]")

# 90 ms
print("90 ms: ")
print("D =", np.median(D_90),"[", np.quantile(D_90, 0.25),",", np.quantile(D_90, 0.75), "]")
print("f =", np.median(f_90),"[", np.quantile(f_90, 0.25),",", np.quantile(f_90, 0.75), "]")
print("D* =", np.median(pD_90),"[", np.quantile(pD_90, 0.25),",", np.quantile(pD_90, 0.75), "]")


# Statistical tests
print("\nStatistical tests")
# D
if stats.shapiro(D_45)[1]>0.05 and stats.shapiro(D_60)[1]>0.05 and stats.shapiro(D_75)[1]>0.05 and stats.shapiro(D_90)[1]>0.05:
    # One-way ANOVA test
    p_D = stats.f_oneway(D_45, D_60, D_75, D_90)[1]
    print("p(D) =", p_D, "; One-way ANOVA")
else:
    # Kruskal-Wallis test
    p_D = stats.kruskal(D_45, D_60, D_75, D_90)[1]
    print("p(D) =", p_D, "; Kruskal-Wallis")

# f
if stats.shapiro(f_45)[1]>0.05 and stats.shapiro(f_60)[1]>0.05 and stats.shapiro(f_75)[1]>0.05 and stats.shapiro(f_90)[1]>0.05:
    # One-way ANOVA test
    p_f = stats.f_oneway(f_45, f_60, f_75, f_90)[1]
    print("p(f) =", p_f, "; One-way ANOVA")
else:
    # Kruskal-Wallis test
    p_f = stats.kruskal(f_45, f_60, f_75, f_90)[1]
    print("p(f) =", p_f, "; Kruskal-Wallis")
    
# D*
if stats.shapiro(pD_45)[1]>0.05 and stats.shapiro(pD_60)[1]>0.05 and stats.shapiro(pD_75)[1]>0.05 and stats.shapiro(pD_90)[1]>0.05:
    # One-way ANOVA test
    p_pD = stats.f_oneway(pD_45, pD_60, pD_75, pD_90)[1]
    print("p(D*) =", p_pD, "; One-way ANOVA")
else:
    # Kruskal-Wallis test
    p_pD = stats.kruskal(pD_45, pD_60, pD_75, pD_90)[1]
    print("p(D*) =", p_pD, "; Kruskal-Wallis")


# Multiple comparison (either Tukey's test or Dunn Sidak correction)
D = np.array([D_45, D_60, D_75, D_90])
multi_D = stats.tukey_hsd(D_45, D_60, D_75,D_90)
# multi_D = sp.posthoc_dunn(D)
print("\nPost-hoc test for D\n")
print(multi_D)


# Number of measurement values
n_meas = np.size(D_45)

# Plot commands

# define flierstyle
flierprops = dict(marker='+', markeredgecolor='red', markersize=20,
                  markeredgewidth=5)  

# font
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 3

# Plots for 4 echo times
fig = plt.figure(figsize=(13, 12))

# D
# fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(2,2,1)  
props = dict(linewidth=3)
ax.boxplot(list([D_45, D_60, D_75, D_90]), 
            whiskerprops=props, boxprops=props, capprops=props, 
            showfliers=True, medianprops=dict(linewidth=3, color='r'),
            flierprops=flierprops, widths=(0.5, 0.5, 0.5, 0.5))
ax.set_ylabel('D (µm²/ms)', fontsize=25)
#ax.set_ylim((7, 43))
xticklabels = ['45 ms', '60 ms', '75 ms', '90 ms']
ax.set_xticklabels(xticklabels)
ax.tick_params(labelsize=25, width=3)

# make random scattering in boxplot
x_45 = np.random.uniform(low=0.85, high=1.15, size=n_meas)
x_60 = np.random.uniform(low=1.85, high=2.15, size=n_meas)
x_75 = np.random.uniform(low=2.85, high=3.15, size=n_meas)
x_90 = np.random.uniform(low=3.85, high=4.15, size=n_meas)
ax.scatter(x_45, D_45, marker='o', s=500, alpha=.5)
ax.scatter(x_60, D_60, marker='o', s=500, alpha=.5, color='pink')
ax.scatter(x_75, D_75, marker='o', s=500, alpha=.5)
ax.scatter(x_90, D_90, marker='o', s=500, alpha=.5)

# Plot mean
ax.plot(1, np.mean(D_45), marker='X', markersize=25, color='darkred')
ax.plot(2, np.mean(D_60), marker='X', markersize=25, color='darkred')
ax.plot(3, np.mean(D_75), marker='X', markersize=25, color='darkred')
ax.plot(4, np.mean(D_90), marker='X', markersize=25, color='darkred')


# f
# fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(2,2,3)  
props = dict(linewidth=3)
ax.boxplot(list([f_45, f_60, f_75, f_90]), 
            whiskerprops=props, boxprops=props, capprops=props, 
            showfliers=True, medianprops=dict(linewidth=3, color='r'),
            flierprops=flierprops, widths=(0.5, 0.5, 0.5, 0.5))
ax.set_ylabel('f (%)', fontsize=25)
#ax.set_ylim((7, 43))
xticklabels = ['45 ms', '60 ms', '75 ms', '90 ms']
ax.set_xticklabels(xticklabels)
ax.tick_params(labelsize=25, width=3)

# make random scattering in boxplot
x_45 = np.random.uniform(low=0.85, high=1.15, size=n_meas)
x_60 = np.random.uniform(low=1.85, high=2.15, size=n_meas)
x_75 = np.random.uniform(low=2.85, high=3.15, size=n_meas)
x_90 = np.random.uniform(low=3.85, high=4.15, size=n_meas)
ax.scatter(x_45, f_45, marker='o', s=500, alpha=.5)
ax.scatter(x_60, f_60, marker='o', s=500, alpha=.5, color='pink')
ax.scatter(x_75, f_75, marker='o', s=500, alpha=.5)
ax.scatter(x_90, f_90, marker='o', s=500, alpha=.5)

# Plot mean
ax.plot(1, np.mean(f_45), marker='X', markersize=25, color='darkred')
ax.plot(2, np.mean(f_60), marker='X', markersize=25, color='darkred')
ax.plot(3, np.mean(f_75), marker='X', markersize=25, color='darkred')
ax.plot(4, np.mean(f_90), marker='X', markersize=25, color='darkred')


# D*
# fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(2,2,2)  
props = dict(linewidth=3)
ax.boxplot(list([pD_45, pD_60, pD_75, pD_90]), 
            whiskerprops=props, boxprops=props, capprops=props, 
            showfliers=True, medianprops=dict(linewidth=3, color='r'),
            flierprops=flierprops, widths=(0.5, 0.5, 0.5, 0.5))
ax.set_ylabel('$D^*$ (µm²/ms)', fontsize=25)
xticklabels = ['45 ms', '60 ms', '75 ms', '90 ms']
ax.set_xticklabels(xticklabels)
ax.tick_params(labelsize=25, width=3)

# make random scattering in boxplot
x_45 = np.random.uniform(low=0.85, high=1.15, size=n_meas)
x_60 = np.random.uniform(low=1.85, high=2.15, size=n_meas)
x_75 = np.random.uniform(low=2.85, high=3.15, size=n_meas)
x_90 = np.random.uniform(low=3.85, high=4.15, size=n_meas)
ax.scatter(x_45, pD_45, marker='o', s=500, alpha=.5)
ax.scatter(x_60, pD_60, marker='o', s=500, alpha=.5, color='pink')
ax.scatter(x_75, pD_75, marker='o', s=500, alpha=.5)
ax.scatter(x_90, pD_90, marker='o', s=500, alpha=.5)

# Plot mean
ax.plot(1, np.mean(pD_45), marker='X', markersize=25, color='darkred')
ax.plot(2, np.mean(pD_60), marker='X', markersize=25, color='darkred')
ax.plot(3, np.mean(pD_75), marker='X', markersize=25, color='darkred')
ax.plot(4, np.mean(pD_90), marker='X', markersize=25, color='darkred')

plt.tight_layout(h_pad=4)  
plt.show()