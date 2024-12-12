import numpy as np
import matplotlib.pyplot as plt


def compartement(f, T2, T1, TE, TR):
    """signal for single compartement"""
    return f*np.exp(-TE/T2)*(1-np.exp(-TR/T1))

def signal(f_k, f_b, f_tub, T2_k, T2_b, T2_tub, T1_k, T1_b, T1_tub, TE, TR):
    """total signal of all three compartements"""
    return compartement(f_k, T2_k, T1_k, TE, TR) + compartement(f_b, T2_b, T1_b, TE, TR) + compartement(f_tub, T2_tub, T1_tub, TE, TR)


# Parameters (time in ms)
TE = np.linspace(45,90,100)
TR = 4500
T1_k, T1_b, T1_tub = 1194, 1600, 4000
T2_k, T2_b, T2_tub = 80, 70, 2000
f_b_A, f_tub_A = 0.207, 0.01
f_b_B, f_tub_B = 0.1, 0.072
f_b_C, f_tub_C = 0, 0.133
f_b_D, f_tub_D = 0.222, 0

f_b_array = np.array([f_b_A, f_b_B, f_b_C, f_b_D])
f_tub_array = np.array([f_tub_A, f_tub_B, f_tub_C, f_tub_D])

fig = plt.figure(figsize=(10,8))

for i, f_b in enumerate(f_b_array):
    f_tub = f_tub_array[i]
    f_k = 1-f_b-f_tub

    S = signal(f_k, f_b, f_tub, T2_k, T2_b, T2_tub, T1_k, T1_b, T1_tub, TE, TR)

    fraction_blood = compartement(f_b, T2_b, T1_b, TE, TR)
    fraction_tub = compartement(f_tub, T2_tub, T1_tub, TE, TR)
    fraction_kidney = compartement(f_k, T2_k, T1_k, TE, TR)

    ax = fig.add_subplot(2,2,i+1)
    
    ax.plot(TE, fraction_blood*100/S, color='red', label='blood', linewidth=3)
    ax.plot(TE, fraction_tub*100/S, color='gold', label='primary urine', linewidth=3)
    ax.plot(TE, (fraction_blood+fraction_tub)*100/S, color='black', label='f_model', linewidth=3, linestyle='dashed')
    # plt.plot(TE, fraction_kidney*100/S, color='blue', label='kidney tissue', linewidth=3)
    ax.set_xlabel('TE (ms)', fontsize=15)
    ax.set_ylabel('signal fraction (%)', fontsize=15)
    ax.set_title('fblood = {}; furine = {}'.format(f_b, f_tub), fontsize=15)
    ax.set_ylim(-1, 24.5)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)

plt.tight_layout()
plt.show()