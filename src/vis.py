import matplotlib.pyplot as plt

import numpy as np

from . import surrogate as srg

def visualize_KL_GP_eigenvectors(KL_GP, exact, z_lims, filename='KL_GP_eigenvectors.png'):

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))

    test_pts = np.linspace(*z_lims, 100)
    phi_GP = KL_GP['phi_GP']

    for phi_i in range(len(phi_GP)):
        plot_row = phi_i // ax.shape[1]
        plot_col = phi_i % ax.shape[1]
    
        var_phi = 3*np.sqrt(phi_GP[phi_i].predict_variances(test_pts))
        mean_phi = phi_GP[phi_i].predict_values(test_pts)

        ax[plot_row, plot_col].plot(test_pts, mean_phi, color='red')
        ax[plot_row, plot_col].plot(test_pts, mean_phi + var_phi, linestyle='--', color='gray')
        ax[plot_row, plot_col].plot(test_pts, mean_phi - var_phi, linestyle='--', color='gray')

        ax[plot_row, plot_col].grid()
        #xi1 = rng.normal()
    ax[0, 0].plot(exact[0, :], exact[1, :])

    plt.savefig(filename, dpi=300)

def visualize_PCE(PCE, lim=None, N=1000, filename='PCE.png'):
    gaussian, model = PCE['gaussian'], PCE['model']
    
    samples = gaussian.sample(N)
    pts = model(*samples)

    fig, ax = plt.subplots(figsize=(5, 5))
    if lim is not None:
        ax.set_xlim(lim)
        ax.hist(pts, bins=10, range=lim)
    else:
        ax.hist(pts, bins=10)
    plt.savefig(filename, dpi=300)

def visualize_KL_GP_distr(KL_GP, z, lim=None, N=1000, filename='KL_GP_distr.png'):
    A, phi, phi_GP, gaussian, expansion = KL_GP['A'], KL_GP['phi'], KL_GP['phi_GP'], KL_GP['gaussian'], KL_GP['expansion']
    
    res = []

    #for i in range(1000):
    #    print(expansion)
    #    res_ = phi_GP[0].predict_values(np.array([z])) * expansion[1] + expansion[2]
    #    print(res_)
    #    xi = gaussian.sample(1)
    #    for k in range(len(phi)):
    #        eta = gaussian.sample(1) 
    #        for j in range(1, A.shape[1]):
    #            res_ = res_ + (A[:, j] @ phi[k])*(phi_GP[k+1].predict_values(np.array([z])) + eta[0]*phi_GP[k+1].predict_variances(np.array([z]))).item()*expansion[j]
    #    print(res_)
    #    res.append(res_)
    
    PCE = phi_GP[0].predict_values(np.array([z])) * expansion[0]
    for k in range(len(phi)):
        eta = gaussian.sample(1) 
        for j in range(1, A.shape[1]):
            PCE = PCE + (A[:, j] @ phi[k])*(phi_GP[k+1].predict_values(np.array([z])) + eta[0]*phi_GP[k+1].predict_variances(np.array([z]))).item()*expansion[j]
    print(PCE)
    res.append(PCE(*gaussian.sample(1000)))
    res = np.array(res).flatten()
    fig, ax = plt.subplots()

    if lim is not None:
        ax.set_xlim(*lim)
        ax.hist(res, bins=10, range=lim)
    else:
        ax.hist(res, bins=10)
    plt.savefig(filename, dpi=300)

# test case specific (for now)
def plot_surrogates(surrogate_1, surrogate_2, xlims, ylims, z, filename='surrogates.png'):
    x_pts = np.linspace(*xlims, 1000)
    y_pts = np.linspace(*ylims, 1000)

    pts_1 = np.dstack((z*np.ones_like(x_pts), x_pts))[0]
    pts_2 = np.dstack((z*np.ones_like(y_pts), y_pts))[0]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.grid()
    
    mean_1 = surrogate_1.predict_values(pts_1)
    var_1 = surrogate_1.predict_variances(pts_1)
    ax.plot(x_pts, mean_1, linestyle='-', color='red')
    ax.plot(x_pts, mean_1 - 3*np.sqrt(var_1), linestyle='--', color='gray')
    ax.plot(x_pts, mean_1 + 3*np.sqrt(var_1), linestyle='--', color='gray')
    
    mean_2 = surrogate_2.predict_values(pts_2)
    var_2 = surrogate_2.predict_variances(pts_2)
    
    ax.plot(mean_2, y_pts, color='blue')
    ax.plot(mean_2 + 3*np.sqrt(var_2), y_pts, linestyle='--', color='black')
    ax.plot(mean_2 - 3*np.sqrt(var_2), y_pts, linestyle='--', color='black')

    res = srg.sample_surrogate_solution([surrogate_1, surrogate_2], z, 500)

    ax.scatter(*np.flip(res.T), marker='x', color='red')
    
    plt.savefig(filename, dpi=300)
    return ax

