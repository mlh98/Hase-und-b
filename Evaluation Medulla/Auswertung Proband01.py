import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from scipy.ndimage import binary_erosion


def load_nifti(path, filename):
    file = os.path.join(path, filename)
    data = nib.load(file)
    data = data.get_fdata()
    data = np.rot90(data, 3, axes=(0, 1))
    data = np.flip(data, axis=1)

    return data


def mask(data, segmentation, is_static=True):
    
    dim_slice_data = np.shape(data)[2]
    dim_slice_seg = np.shape(segmentation)[2]
    
    # if dim_slice_seg-dim_slice_data > 10e-09:
    #     segmentation = segmentation[:, :, 0:dim_slice_data:1, :]
    # else:
    #     pass
    # delete slices that were not segmented
    del_ind = []
    for slice in range(dim_slice_data):
        if np.sum(segmentation[:, :, slice]) < 10e-09:
            del_ind.append(slice)
            
    segmentation = np.delete(segmentation, del_ind, axis=2)
    data = np.delete(data, del_ind, axis=2)
    
    segmentation[segmentation == 0] = np.nan

    # mask
    mask = data*segmentation

    return mask


def get_median(mask):
    return np.nanmedian(mask, axis=(0,1))


def average_diffusion(median, b):
    # sort b-values and images
    sorted_index = b.argsort()
    b = b[sorted_index]
    median = median[sorted_index]
    
    av_ind = [np.where(b == element)[0].tolist() for element in
              np.unique(b)]

    b_unique = np.unique(b)

    # Compute new weights
    weights_unique = np.zeros(len(av_ind))
    i = 0
    for element in av_ind:
        weights_unique[i] = 1/len(element)
        i = i+1

    # Average diffusion
    average = np.zeros(np.size(b_unique))
    for i, ind in enumerate(av_ind):
        average[i] = np.nanmean(median[ind])

    # # Normalize to b0
    # average = average/average[0]

    return average, b_unique, weights_unique


def fit(average, b_unique, weights_unique, mono_exp, bi_exp, tri_exp, logarithmic=True,
        exclude_b0=False):
    
    # Exclude b0
    if exclude_b0==True:
        average = average[1:]
        b_unique = b_unique[1:]
        weights_unique = weights_unique[1:]
    else:
        average = average/average[0]
    
    # seperating data above and below 200
    signal_above = average[np.where(b_unique >= 200)]
    b_above = b_unique[np.where(b_unique >= 200)]
    weights_above = weights_unique[np.where(b_unique >= 200)]
    
    if exclude_b0==True:
        # Initial guess and bounds
        init_guess_mo = np.array([0.002, 0.75*average[0]])
        bounds_mo = ([0.001, 0], [0.005, 1.5*average[0]])
    else:
        # Initial guess and bounds
        init_guess_mo = np.array([0.002, 0.75])
        bounds_mo = ([0.001, 0], [0.005, 1.5])
        
    # Fitting above 200
    popt_above, pcov_above = curve_fit(f=mono_exp, xdata=b_above,
                                       ydata=signal_above, p0=init_guess_mo,
                                       bounds=bounds_mo, sigma=weights_above)
    D = popt_above[0]          # Diffusion coefficient
    S0_mo = popt_above[1]
    
    if exclude_b0==True:
        f_estimate = 1-S0_mo/average[0]
    else:
        f_estimate = 1-S0_mo

    if exclude_b0==True: 
        if f_estimate>10e-09:
            # Biexponential fit
            # Initial guess for D, f, D* and S0_bi
            init_guess_bi = np.array([D, f_estimate, 0.1, average[0]])
            bounds_bi = ([D, 0.9999*f_estimate, 0.005, 0.5*average[0]], [D*1.001, 1.0001*f_estimate, 5, 1.5*average[0]])
        else:
            init_guess_bi = np.array([D, 0.2, 0.1, average[0]])
            bounds_bi = ([D, 0, 0.005, 0.5*average[0]], [D*1.001, 0.5, 5, 1.5*average[0]])
    else:
        if f_estimate>10e-09:
            # Biexponential fit
            # Initial guess for D, f, D* and S0_bi
            init_guess_bi = np.array([D, f_estimate, 0.1, 1])
            bounds_bi = ([D, 0.9999*f_estimate, 0.005, 0.99999], [D*1.001, 1.0001*f_estimate, 5, 1.000001])
        else:
            init_guess_bi = np.array([D, 0.2, 0.1, 1])
            bounds_bi = ([D, 0, 0.005, 0.99999], [D*1.001, 0.5, 5, 1.000001])
            
    popt_bi, pcov_bi = curve_fit(f=bi_exp, xdata=b_unique, sigma=weights_unique, 
                                 ydata=average, p0=init_guess_bi, 
                                 bounds=bounds_bi)
    D = popt_bi[0]
    f = popt_bi[1]
    pD = popt_bi[2]
    S0_bi = popt_bi[3]

    res_bi = np.array([D, f, pD, S0_bi])
    
    
    if exclude_b0==True:
        if f_estimate>10e-09:
            # Triexponential Fit
            # Initial guess
            init_guess_tri = np.array([D, f_estimate*0.7, f_estimate*0.7, 0.03, 1, average[0]])
            # Bounds
            bounds_tri = ([D*0.99999, 0, 0, 0.005, 0.15, 0.5*average[0]], [D*1.00001, 1.5*f_estimate, 1.5*f_estimate, 0.3, 20, 1.5*average[0]])
        else:
            init_guess_tri = np.array([D, 0.1, 0.1, 0.03, 1, average[0]])
            # Bounds
            bounds_tri = ([D*0.99999, 0, 0, 0.005, 0.15, 0.5*average[0]], [D*1.00001, 0.5, 0.5, 0.3, 20, 1.5*average[0]])
    else:
        if f_estimate>10e-09:
            # Triexponential Fit
            # Initial guess
            init_guess_tri = np.array([D, f_estimate*0.7, f_estimate*0.7, 0.03, 1, 1])
            # Bounds
            bounds_tri = ([D*0.99999, 0, 0, 0.005, 0.15, 0.9999999], [D*1.00001, 1.5*f_estimate, 1.5*f_estimate, 0.3, 20, 1.000001])
        else:
            # Initial guess
            init_guess_tri = np.array([D, 0.1, 0.1, 0.03, 1, 1])
            # Bounds
            bounds_tri = ([D*0.99999, 0, 0, 0.005, 0.15, 0.99999], [D*1.00001, 0.5, 0.5, 0.3, 20, 1.00001])
            
    popt_tri, pcov_tri = curve_fit(f=tri_exp, xdata=b_unique, sigma=weights_unique, 
                                 ydata=average, p0=init_guess_tri, 
                                 bounds=bounds_tri)

    f1 = popt_tri[1]
    f2 = popt_tri[2]
    pD1 = popt_tri[3]
    pD2 = popt_tri[4]
    S0_tri = popt_tri[5]
    
    res_tri = np.array([D, f1, f2, pD1, pD2, S0_tri])

    # Plotting data and fit
    b_fit = np.linspace(0, 900, 1000)
    fit_mono = mono_exp(b_fit, D, S0_mo)
    fit_bi = bi_exp(b_fit, D, f, pD, S0_bi)
    fit_tri = tri_exp(b_fit, D, f1, f2, pD1, pD2, S0_tri)

    # Plot
    # font
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot data and find
    ax.plot(b_unique, average, marker='o', markersize=15,
            linestyle='None', color='blue', fillstyle='none', label='Data')
    ax.plot(b_fit, fit_mono, color='blue', label='monoexponential fit', 
            linewidth=2.5, linestyle='dashed')
    ax.plot(b_fit, fit_bi, color='firebrick', label='biexponential fit', linewidth=2.5)
    ax.plot(b_fit, fit_tri, color='orangered', label='triexponential fit', linewidth=2.5)

    # Labeling
    ax.set_xlabel("b [$\mathrm{s/mm^2}$]", fontsize=20)
    ax.set_ylabel("$S/S(b=0)$", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    # ax.legend(fontsize=20)
    # annotation = "D = {:.3f}\nf = {:.3f}\nD* = {:.3f}".format(D*1000, f*100, pD*1000)
    # ax.annotate(annotation, xy=(600, 0.4))
    
    if exclude_b0==False:
        # ax.set_xlim(-5, 150)
        # ax.set_ylim(0.5, 1.05)
        ax.set_ylim(0.13, 1.1)
    else:
        ax.set_xlim(-5, 100)
        ax.set_ylim(0.67*S0_tri, 1.2*S0_tri)

    if logarithmic == True:
        plt.yscale('log')

    # make tick labels decimal numbers
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                      pos: str(int(round(x)))))
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

    # make small tick labels on y axis larger
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(20)

    plt.show()

    return res_bi, res_tri


def save_results(path_nifti, filename_nifti_b1_list, filename_nifti_b2_list, 
                 path_segmentation, filename_seg_b1_list, filename_seg_b2_list, 
                 slicewise, slices, b, file_save, path_save_list_bi, path_save_list_tri, logarithmic, exclude_b0):
    
    
    for i, file_b1 in enumerate(filename_nifti_b1_list): 
        data_b1 = load_nifti(path_nifti, file_b1)
        data_b2 = load_nifti(path_nifti, filename_nifti_b2_list[i])
        data = np.append(data_b1, data_b2, axis=3)   
                
        segmentation_b1 = load_nifti(path_segmentation, filename_seg_b1_list[i])
        segmentation_b2 = load_nifti(path_segmentation, filename_seg_b2_list[i])
        
        if segmentation_b1.ndim < 4:
            segmentation_b1 = np.expand_dims(segmentation_b1, axis=-1)  
            segmentation_b2 = np.expand_dims(segmentation_b2, axis=-1)  
            segmentation_b1 = segmentation_b1*np.ones_like(data_b1)
            segmentation_b2 = segmentation_b2*np.ones_like(data_b2)
            
            segmentation = np.append(segmentation_b1, segmentation_b2, axis=3)
            
        else:
            segmentation = np.append(segmentation_b1, segmentation_b2, axis=3)
            
        # Remove cortex from segmentation
        cortex_thickness = 3
        for slice_ind in slices:
            for b_ind in range(segmentation.shape[3]):
                segmentation_medulla = binary_erosion(segmentation[:,:,slice_ind,b_ind], iterations=cortex_thickness)    
                
                # Check if segmentation still contains values
                # Otherwise decrease cortex thickness
                loop=1
                while np.max(segmentation_medulla)<1 and loop<cortex_thickness:
                    segmentation_medulla = binary_erosion(segmentation[:,:,slice_ind,b_ind], iterations=cortex_thickness-loop)
                    loop=loop+1
                    if np.max(segmentation_medulla)<1 and cortex_thickness-loop<10e-09:
                        segmentation_medulla = segmentation[:,:,slice_ind,b_ind]
                segmentation[:,:,slice_ind,b_ind] = segmentation_medulla 
                
        masked_values = mask(data, segmentation)
        median = get_median(masked_values)
        
        if slicewise==True:
            res_bi = np.zeros((len(slices) ,4))
            res_tri = np.zeros((len(slices), 6))
            for slice_ind in slices:
                # average over b-values
                average, b_unique, weights_unique = average_diffusion(median[slice_ind,:], b)
                # perform fit
                bi_array, tri_array = fit(average, b_unique, weights_unique, 
                                          mono_exp, bi_exp, tri_exp, logarithmic, exclude_b0)
                res_bi[slice_ind,:] = bi_array[:]
                res_tri[slice_ind,:] = tri_array[:]
                
        elif slicewise==False:
            res_bi = np.zeros(4)
            res_tri = np.zeros(6)
            # Average over slices
            median = np.median(median, axis=0)
            
            average, b_unique, weights_unique = average_diffusion(median, b)
            # perform fit
            res_bi, res_tri = fit(average, b_unique, weights_unique, mono_exp, bi_exp, 
                                  tri_exp, logarithmic, exclude_b0)
            
        else:
            print('Enter slicewise "True" or "False".')
            
        
        # Save bi- and triexponential
        header_bi = "D, f, D*, S0"
        header_tri = "D, f1, f2, D1*, D2*, S0"
        np.savetxt(os.path.join(path_save_list_bi[i], file_save), res_bi, 
                    header=header_bi)
        np.savetxt(os.path.join(path_save_list_tri[i], file_save), res_tri, 
                    header=header_tri)

    return res_bi, res_tri


# Define fit functions
def mono_exp(b, D, S0):
    return S0*np.exp(-b*D)

def bi_exp(b, D, f, pD, S0):
    return S0*((1-f)*np.exp(-b*D) + f*np.exp(-b*pD))

def tri_exp(b, D, f1, f2, pD1, pD2, S0):
    return S0*((1-f1-f2)*np.exp(-b*D) + f1*np.exp(-b*pD1) + f2*np.exp(-b*pD2))

# Path and files

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of script_dir
parent_dir = os.path.dirname(script_dir)

# Construct the path to measurement folder
meas_dir = os.path.join(parent_dir, 'Messungen und Segmentierung')

# Construct the path to the 'Ergebnisse' folder
results_dir = os.path.join(parent_dir, 'Ergebnisse')

# Nifti and segmentation
path_nifti = os.path.join(meas_dir, '01', 'data')
path_segmentation = os.path.join(meas_dir, '01', 'segmentation')

# for a whole list of files
filename_nifti_b1_list = ['45_b1.nii', '60_b1.nii', '75_b1.nii', '90_b1.nii']
filename_nifti_b2_list = ['45_b2.nii', '60_b2.nii', '75_b2.nii', '90_b2.nii']

filename_seg_b1_list_r = ['seg_45_b1_r.nii', 'seg_60_b1_r.nii', 
                          'seg_75_b1_r.nii', 'seg_90_b1_r.nii']
filename_seg_b2_list_r = ['seg_45_b2_r.nii', 'seg_60_b2_r.nii', 
                          'seg_75_b2_r.nii', 'seg_90_b2_r.nii']
filename_seg_b1_list_l = ['seg_45_b1_l.nii', 'seg_60_b1_l.nii', 
                          'seg_75_b1_l.nii', 'seg_90_b1_l.nii']
filename_seg_b2_list_l = ['seg_45_b2_l.nii', 'seg_60_b2_l.nii', 
                          'seg_75_b2_l.nii', 'seg_90_b2_l.nii']

# Paths for saving results (biexponential)
path_save_45_l_bi = os.path.join(results_dir, 'Medulla', 'links', '45_bi')
path_save_60_l_bi = os.path.join(results_dir, 'Medulla', 'links', '60_bi')
path_save_75_l_bi = os.path.join(results_dir, 'Medulla', 'links', '75_bi')
path_save_90_l_bi = os.path.join(results_dir, 'Medulla', 'links', '90_bi')

path_save_45_r_bi = os.path.join(results_dir, 'Medulla', 'rechts', '45_bi')
path_save_60_r_bi = os.path.join(results_dir, 'Medulla', 'rechts', '60_bi')
path_save_75_r_bi = os.path.join(results_dir, 'Medulla', 'rechts', '75_bi')
path_save_90_r_bi = os.path.join(results_dir, 'Medulla', 'rechts', '90_bi')

# Paths for saving results (triexponential)
path_save_45_l_tri = os.path.join(results_dir, 'Medulla', 'links', '45_tri')
path_save_60_l_tri = os.path.join(results_dir, 'Medulla', 'links', '60_tri')
path_save_75_l_tri = os.path.join(results_dir, 'Medulla', 'links', '75_tri')
path_save_90_l_tri = os.path.join(results_dir, 'Medulla', 'links', '90_tri')

path_save_45_r_tri = os.path.join(results_dir, 'Medulla', 'rechts', '45_tri')
path_save_60_r_tri = os.path.join(results_dir, 'Medulla', 'rechts', '60_tri')
path_save_75_r_tri = os.path.join(results_dir, 'Medulla', 'rechts', '75_tri')
path_save_90_r_tri = os.path.join(results_dir, 'Medulla', 'rechts', '90_tri')

path_save_list_r_bi = [path_save_45_r_bi, path_save_60_r_bi, path_save_75_r_bi, 
                    path_save_90_r_bi]
path_save_list_l_bi = [path_save_45_l_bi, path_save_60_l_bi, path_save_75_l_bi, 
                    path_save_90_l_bi]

path_save_list_r_tri = [path_save_45_r_tri, path_save_60_r_tri, path_save_75_r_tri, 
                    path_save_90_r_tri]
path_save_list_l_tri = [path_save_45_l_tri, path_save_60_l_tri, path_save_75_l_tri, 
                    path_save_90_l_tri]

file_save = '01.txt'

b = np.array([0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 70, 70, 70, 70, 200,
              200, 200, 200, 800, 800, 800, 800, 70, 70, 70, 70, 1, 1, 1, 1,
              3.5, 3.5, 3.5, 3.5, 5, 5, 5, 5, 70, 70, 70, 70, 1.2, 1.2, 1.2,
              1.2, 6, 6, 6, 6, 45, 45, 45, 45, 1.5, 1.5, 1.5, 1.5, 60, 60, 60,
              60, 1.8, 1.8, 1.8, 1.8, 10, 10, 10, 10, 700, 700, 700, 700, 0.2,
              0.2, 0.2, 0.2, 2, 2, 2, 2, 25, 25, 25, 25, 6, 6, 6, 6, 35, 35,
              35, 35])

slicewise=True
logarithmic=True
exclude_b0=False
slices = np.array([0, 1, 2, 3])   # slice indices
print(file_save)

# left kidney
res_bi, res_tri = save_results(path_nifti, filename_nifti_b1_list, 
                               filename_nifti_b2_list, path_segmentation, 
                               filename_seg_b1_list_l, filename_seg_b2_list_l, slicewise, 
                               slices, b, file_save, path_save_list_l_bi, path_save_list_l_tri, logarithmic, exclude_b0)

# right kidney
res_bi, res_tri = save_results(path_nifti, filename_nifti_b1_list, 
                               filename_nifti_b2_list, path_segmentation, 
                               filename_seg_b1_list_r, filename_seg_b2_list_r, slicewise, 
                               slices, b, file_save, path_save_list_r_bi, path_save_list_r_tri, logarithmic, exclude_b0)