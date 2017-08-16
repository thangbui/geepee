import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
import sys
sys.path.append('../../../')
import geepee.aep_models as aep
from geepee.config import PROP_MC, PROP_MM
np.random.seed(42)
import pdb


def find_interval(arr, val):
    intervals = []
    found = False
    start_idx = 0
    end_idx = 0
    for i in range(arr.shape[0]):
        if (not found) and arr[i] == val:
            found = True
            start_idx = i
        elif found and arr[i] != val:
            found = False
            end_idx = i
            intervals.append([start_idx, end_idx])
    if found:
        intervals.append([start_idx, arr.shape[0]])
    return intervals


def plot_model_with_control(model, plot_title='', name_suffix=''):
    # plot function
    mx, vx = model.get_posterior_x()
    mins = np.min(mx, axis=0) - 0.5
    maxs = np.max(mx, axis=0) + 0.5
    nGrid = 50
    xspaced = np.linspace(mins[0], maxs[0], nGrid)
    yspaced = np.linspace(mins[1], maxs[1], nGrid)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T

    x_control = model.x_control
    c_unique = np.unique(x_control)
    colors = ['g', 'b', 'r', 'm']
    for c_ind, c_val in enumerate(c_unique):
        idxs = np.where(x_control == c_val)[0]
        X_c_pad = c_val * np.ones((Xplot.shape[0], 1))
        Xplot_c = np.hstack((Xplot, X_c_pad))
        intervals = find_interval(x_control, c_val)
        # mins = np.min(mx[idxs, :], axis=0)-0.5
        # maxs = np.max(mx[idxs, :], axis=0)+0.5

        mf, vf = model.predict_f(Xplot_c)
        fig = plt.figure()
        plt.imshow((mf[:, 0]).reshape(*xx.shape),
                   vmin=mf.min(), vmax=mf.max(), origin='lower',
                   extent=[mins[0], maxs[0], mins[1], maxs[1]], aspect='auto', alpha=0.4, cmap=plt.cm.viridis)
        plt.colorbar()
        plt.contour(
            xx, yy, (mf[:, 0]).reshape(*xx.shape),
            colors='k', linewidths=2, zorder=100)
        zu = model.dyn_layer.zu
        plt.plot(zu[:, 0], zu[:, 1], 'ko', mew=0, ms=4)
        # plt.plot(mx[idxs, 0], mx[idxs, 1], 'bo', ms=3, zorder=101)
        for k, inter in enumerate(intervals):
            idxs = np.arange(inter[0], inter[1])
            for i in range(inter[0], inter[1] - 1):
                plt.plot(mx[i:i + 2, 0], mx[i:i + 2, 1],
                         '-o', color=colors[k], ms=3, linewidth=1, zorder=101)
        plt.xlabel(r'$x_{t, 1}$')
        plt.ylabel(r'$x_{t, 2}$')
        plt.xlim([mins[0], maxs[0]])
        plt.ylim([mins[1], maxs[1]])
        plt.title(plot_title)
        plt.savefig('/tmp/hh_gpssm_dim_0' +
                    name_suffix + '_c_%.2f.pdf' % c_val)

        fig = plt.figure()
        plt.imshow((mf[:, 1]).reshape(*xx.shape),
                   vmin=mf.min(), vmax=mf.max(), origin='lower',
                   extent=[mins[0], maxs[0], mins[1], maxs[1]], aspect='auto', alpha=0.4, cmap=plt.cm.viridis)
        plt.colorbar()
        plt.contour(
            xx, yy, (mf[:, 1]).reshape(*xx.shape),
            colors='k', linewidths=2, zorder=100)
        zu = model.dyn_layer.zu
        plt.plot(zu[:, 0], zu[:, 1], 'ko', mew=0, ms=4)
        # plt.plot(mx[idxs, 0], mx[idxs, 1], 'bo', ms=3, zorder=101)
        for k, inter in enumerate(intervals):
            idxs = np.arange(inter[0], inter[1])
            for i in range(inter[0], inter[1] - 1):
                plt.plot(mx[i:i + 2, 0], mx[i:i + 2, 1],
                         '-o', color=colors[k], ms=3, linewidth=1, zorder=101)
        # plt.xlabel(r'$x_{t, 1}$')
        # plt.ylabel(r'$x_{t, 2}$')
        plt.xlim([mins[0], maxs[0]])
        plt.ylim([mins[1], maxs[1]])
        plt.title(plot_title)
        plt.savefig('/tmp/hh_gpssm_dim_1' +
                    name_suffix + '_c_%.2f.pdf' % c_val)


def plot_posterior_gp(data_fname, params_fname, fig_fname, M=20):
    # load dataset
    data = np.loadtxt(data_fname)
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = xc
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=x_control, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    my, vy, vyn = model_aep.get_posterior_y()
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy[:, i]
        vin = vyn[:, i]
        plt.subplot(no_panes, 1, i + 1)
        plt.fill_between(t, mi + 2 * np.sqrt(vi), mi - 2 *
                         np.sqrt(vi), color=cs[i], alpha=0.4)
        plt.plot(t, mi, '-', color=cs[i])
        plt.plot(t, yi, '--', color=cs[i])
        plt.ylabel(labels[i])
        plt.xticks([])
        plt.yticks([])

    plt.subplot(no_panes, 1, no_panes)
    plt.plot(t, x_control, '-', color='m')
    plt.ylabel('I')
    # plt.yticks([])
    plt.xlabel('t')

    plt.savefig(fig_fname)

    # plot_model_with_control(model_aep, '', '_gp_with_control')


def plot_prediction_gp_MM(data_fname, params_fname, fig_fname, M=20):
    # load dataset
    data = np.loadtxt(data_fname)
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = xc
    x_control_test = np.flipud(x_control)
    # x_control_test = x_control * 1.5
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=x_control, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    print 'ls ', np.exp(model_aep.dyn_layer.ls)
    my, vy, vyn = model_aep.get_posterior_y()
    mxp, vxp, myp, vyp, vynp = model_aep.predict_forward(T, x_control_test, prop_mode=PROP_MM)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    lims = [[-6, 3], [-1, 6], [1, 7], [-1, 5]]
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy[:, i]
        vin = vyn[:, i]
        mip = myp[:, i]
        vip = vyp[:, i]
        vinp = vynp[:, i]
        vip = vinp

        plt.subplot(5, 1, i + 1)
        plt.fill_between(t, mi + 2 * np.sqrt(vi), mi - 2 *
                         np.sqrt(vi), color=cs[i], alpha=0.4)
        plt.plot(t, mi, '-', color=cs[i])
        plt.fill_between(np.arange(T, 2 * T), mip + 2 * np.sqrt(vip),
                         mip - 2 * np.sqrt(vip), color=cs[i], alpha=0.4)
        plt.plot(np.arange(T, 2 * T), mip, '-', color=cs[i])
        plt.plot(t, yi, '--', color=cs[i])
        plt.axvline(x=T, color='k', linewidth=2)
        plt.ylabel(labels[i])
        plt.xticks([])
        plt.ylim(lims[i])
        plt.yticks([])

    plt.subplot(no_panes, 1, no_panes)
    plt.plot(t, x_control, '-', color='m')
    plt.plot(np.arange(T, 2 * T), x_control_test, '-', color='m')
    plt.axvline(x=T, color='k', linewidth=2)
    plt.ylabel('I')
    # plt.yticks([])
    plt.xlabel('t')
    plt.savefig(fig_fname)


def plot_prediction_gp_MC(data_fname, params_fname, fig_fname, M=20, no_samples=100):
    # TODO
    # load dataset
    data = np.loadtxt(data_fname)
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = xc
    x_control_test = np.flipud(x_control)
    # x_control_test = x_control * 1.5
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=x_control, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    print 'ls ', np.exp(model_aep.dyn_layer.ls)
    my, vy, vyn = model_aep.get_posterior_y()
    _, my_MC, vy_MC = model_aep.predict_forward(T, x_control_test, prop_mode=PROP_MC, no_samples=no_samples)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    lims = [[-6, 3], [-1, 6], [1, 7], [-1, 5]]
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy[:, i]
        vin = vyn[:, i]

        mip = my_MC[:, :, i]
        vip = vy_MC[:, :, i]
        
        plt.subplot(no_panes, 1, i + 1)
        plt.fill_between(t, mi + 2 * np.sqrt(vi), mi - 2 *
                         np.sqrt(vi), color=cs[i], alpha=0.4)
        plt.plot(t, mi, '-', color=cs[i])
        plt.plot(t, yi, '--', color=cs[i])
        
        # for k in range(mip.shape[1]):
        for k in range(20):
            plt.plot(np.arange(T, 2*T), mip[:, k], color=cs[i], alpha=0.1)

        plt.axvline(x=T, color='k', linewidth=2)
        plt.ylabel(labels[i])
        plt.xticks([])
        plt.ylim(lims[i])
        plt.yticks([])

    plt.subplot(no_panes, 1, no_panes)
    plt.plot(t, x_control, '-', color='m')
    plt.plot(np.arange(T, 2 * T), x_control_test, '-', color='m')
    plt.axvline(x=T, color='k', linewidth=2)
    plt.ylabel('I')
    # plt.yticks([])
    plt.xlabel('t')
    plt.savefig(fig_fname)


def plot_prediction_gp_MM_fixed_function(params_fname, fig_fname, M=20):
    # load dataset
    data = np.loadtxt('hh_data.txt')
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = xc
    # x_control_test = np.flipud(x_control)
    x_control_test = x_control * 1.5
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=x_control, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    print 'ls ', np.exp(model_aep.dyn_layer.ls)
    my, vy, vyn = model_aep.get_posterior_y()
    mxp, vxp, myp, vyp, vynp = model_aep.predict_forward_fixed_function(T, x_control_test, prop_mode=PROP_MM)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    lims = [[-6, 3], [-1, 6], [1, 7], [-1, 5]]
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy[:, i]
        vin = vyn[:, i]
        mip = myp[:, i]
        vip = vyp[:, i]
        vinp = vynp[:, i]
        vip = vinp

        plt.subplot(5, 1, i + 1)
        plt.fill_between(t, mi + 2 * np.sqrt(vi), mi - 2 *
                         np.sqrt(vi), color=cs[i], alpha=0.4)
        plt.plot(t, mi, '-', color=cs[i])
        plt.fill_between(np.arange(T, 2 * T), mip + 2 * np.sqrt(vip),
                         mip - 2 * np.sqrt(vip), color=cs[i], alpha=0.4)
        plt.plot(np.arange(T, 2 * T), mip, '-', color=cs[i])
        plt.plot(t, yi, '--', color=cs[i])
        plt.axvline(x=T, color='k', linewidth=2)
        plt.ylabel(labels[i])
        plt.xticks([])
        plt.ylim(lims[i])
        plt.yticks([])

    plt.subplot(no_panes, 1, no_panes)
    plt.plot(t, x_control, '-', color='m')
    plt.plot(np.arange(T, 2 * T), x_control_test, '-', color='m')
    plt.axvline(x=T, color='k', linewidth=2)
    plt.ylabel('I')
    plt.yticks([])
    plt.xlabel('t')
    plt.savefig(fig_fname)


def plot_prediction_gp_MC_fixed_function(params_fname, fig_fname, M=20, no_samples=100):
    # TODO
    # load dataset
    data = np.loadtxt('hh_data.txt')
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = xc
    # x_control_test = np.flipud(x_control)
    x_control_test = x_control * 1.5
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=x_control, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    print 'ls ', np.exp(model_aep.dyn_layer.ls)
    my, vy, vyn = model_aep.get_posterior_y()
    _, my_MC, vy_MC = model_aep.predict_forward_fixed_function(
        T, x_control_test, prop_mode=PROP_MC, no_samples=no_samples)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    lims = [[-6, 3], [-1, 6], [1, 7], [-1, 5]]
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy[:, i]
        vin = vyn[:, i]

        mip = my_MC[:, :, i]
        vip = vy_MC[:, :, i]
        
        plt.subplot(no_panes, 1, i + 1)
        plt.fill_between(t, mi + 2 * np.sqrt(vi), mi - 2 *
                         np.sqrt(vi), color=cs[i], alpha=0.4)
        plt.plot(t, mi, '-', color=cs[i])
        plt.plot(t, yi, '--', color=cs[i])
        
        # for k in range(mip.shape[1]):
        for k in range(20):
            plt.plot(np.arange(T, 2*T), mip[:, k], color=cs[i], alpha=0.1)

        plt.axvline(x=T, color='k', linewidth=2)
        plt.ylabel(labels[i])
        plt.xticks([])
        plt.ylim(lims[i])
        plt.yticks([])

    plt.subplot(no_panes, 1, no_panes)
    plt.plot(t, x_control, '-', color='m')
    plt.plot(np.arange(T, 2 * T), x_control_test, '-', color='m')
    plt.axvline(x=T, color='k', linewidth=2)
    plt.ylabel('I')
    plt.yticks([])
    plt.xlabel('t')
    plt.savefig(fig_fname)


def plot_prediction_gp_MM_fixed_function_fixed_control(params_fname, fig_fname, cval, Tcontrol, M=20, prior=False):
    # load dataset
    data = np.loadtxt('hh_data.txt')
    # use the voltage and potasisum current
    std_data = np.std(data, axis=0)
    data = data / std_data
    y = data[:, :4]
    xc = data[:, [-1]]
    cval = cval / std_data[-1]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = cval * np.ones([Tcontrol, 1])
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=xc, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    K_samples = 5
    fig, axs = plt.subplots(5, K_samples+1, figsize=(12, 8))
    lims = [[-6, 3], [-1, 6], [1, 7], [-1, 5], [-5, 35]]
    for k in range(K_samples):
        np.random.seed(k)
        mxp, vxp, myp, vyp, vynp = model_aep.predict_forward_fixed_function(
            Tcontrol, x_control, prop_mode=PROP_MM,
            starting_from_prior=prior)
        for i in range(4):
            mip = myp[:, i]
            vip = vyp[:, i]
            vinp = vynp[:, i]
            vip = vinp

            ax = axs[i, k]
            ax.fill_between(np.arange(Tcontrol), mip + 2 * np.sqrt(vip),
                             mip - 2 * np.sqrt(vip), color=cs[i], alpha=0.4)
            ax.plot(np.arange(Tcontrol), mip, '-', color=cs[i])
            ax.set_ylabel(labels[i])
            ax.set_xticks([])
            ax.set_ylim(lims[i])
            ax.set_yticks([])

        ax = axs[4, k]
        ax.plot(np.arange(Tcontrol), x_control, '-', color='m')
        ax.set_ylabel('I')
        ax.set_yticks([])
        ax.set_xlabel('t')
        ax.set_ylim(lims[-1])

    mxp, vxp, myp, vyp, vynp = model_aep.predict_forward(Tcontrol, x_control, prop_mode=PROP_MM)
    for i in range(4):
        mip = myp[:, i]
        vip = vyp[:, i]
        vinp = vynp[:, i]
        vip = vinp

        ax = axs[i, k+1]
        ax.fill_between(np.arange(Tcontrol), mip + 2 * np.sqrt(vip),
                         mip - 2 * np.sqrt(vip), color=cs[i], alpha=0.4)
        ax.plot(np.arange(Tcontrol), mip, '-', color=cs[i])
        ax.set_ylabel(labels[i])
        ax.set_xticks([])
        ax.set_ylim(lims[i])
        ax.set_yticks([])

    ax = axs[4, k+1]
    ax.plot(np.arange(Tcontrol), x_control, '-', color='m')
    ax.set_ylabel('I')
    ax.set_yticks([])
    ax.set_xlabel('t')
    ax.set_ylim(lims[-1])
    plt.savefig(fig_fname)


def plot_prediction_gp_MC_fixed_function_fixed_control(params_fname, fig_fname, cval, Tcontrol, M=20, prior=False):
    # load dataset
    data = np.loadtxt('hh_data.txt')
    # use the voltage and potasisum current
    std_data = np.std(data, axis=0)
    data = data / std_data
    y = data[:, :4]
    xc = data[:, [-1]]
    cval = cval / std_data[-1]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = cval * np.ones([Tcontrol, 1])
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=xc, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    K_samples = 10
    fig, axs = plt.subplots(5, K_samples+1, figsize=(12, 8))
    lims = [[-6, 3], [-1, 6], [1, 7], [-1, 5], [-5, 35]]
    for k in range(K_samples):
        np.random.seed(k)
        
        _, my_MC, vy_MC = model_aep.predict_forward_fixed_function(
            Tcontrol, x_control, prop_mode=PROP_MC, no_samples=50,
            starting_from_prior=prior)
        for i in range(4):
            mip = my_MC[:, :, i]
            vip = vy_MC[:, :, i]

            ax = axs[i, k]
            for m in range(2):
                ax.plot(np.arange(Tcontrol), mip[:, m], color=cs[i], alpha=0.3)
            ax.set_ylabel(labels[i])
            ax.set_xticks([])
            ax.set_ylim(lims[i])
            ax.set_yticks([])

        ax = axs[4, k]
        ax.plot(np.arange(Tcontrol), x_control, '-', color='m')
        ax.set_ylabel('I')
        ax.set_yticks([])
        ax.set_xlabel('t')
        ax.set_ylim(lims[-1])

    _, my_MC, vy_MC = model_aep.predict_forward(
        Tcontrol, x_control, prop_mode=PROP_MC, no_samples=50)
    for i in range(4):
        mip = my_MC[:, :, i]
        vip = vy_MC[:, :, i]

        ax = axs[i, k+1]
        for m in range(2):
            ax.plot(np.arange(Tcontrol), mip[:, m], color=cs[i], alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.set_xticks([])
        ax.set_ylim(lims[i])
        ax.set_yticks([])

    ax = axs[4, k+1]
    ax.plot(np.arange(Tcontrol), x_control, '-', color='m')
    ax.set_ylabel('I')
    ax.set_yticks([])
    ax.set_xlabel('t')
    ax.set_ylim(lims[-1])
    plt.savefig(fig_fname)


def test_entropy_convergence(params_fname, cval, Tcontrol, M=20, prior=False):
    from mutual_info import entropy
    # load dataset
    data = np.loadtxt('hh_data.txt')
    # use the voltage and potasisum current
    std_data = np.std(data, axis=0)
    data = data / std_data
    y = data[:, :4]
    xc = data[:, [-1]]
    cval = cval / std_data[-1]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = cval * np.ones([Tcontrol, 1])
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=xc, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    no_func_samples = 200
    no_trials = 10
    N_samples = [10, 20, 30, 50, 100, 200, 300, 400, 500]
    for trial in range(no_trials):
        np.random.seed(trial)
        for no_y_samples in N_samples:
            Hy_fixed_func = np.zeros(no_func_samples)
            for k in range(no_func_samples):
                # if k % 50 == 0:
                #     print k, no_func_samples
                _, my_MC, vy_MC = model_aep.predict_forward_fixed_function(
                    Tcontrol, x_control, prop_mode=PROP_MC, no_samples=no_y_samples,
                    starting_from_prior=prior)
                # draw y sample
                # y_samples = np.random.randn(Tcontrol, no_y_samples, 4) * np.sqrt(vy_MC) + my_MC
                y_samples = my_MC
                Hy_fixed_func_k = 0
                for t in range(Tcontrol):
                    # if t % 20 == 0:
                    #     print t, Tcontrol
                    for d in range(4):
                        Hy_fixed_func_k += entropy(y_samples[t, :, d].reshape([no_y_samples, 1]), k=int(np.sqrt(no_y_samples)))
                Hy_fixed_func[k] = Hy_fixed_func_k

            _, my_MC, vy_MC = model_aep.predict_forward(
                Tcontrol, x_control, prop_mode=PROP_MC, no_samples=no_y_samples)
            # draw y sample
            # y_samples = np.random.randn(Tcontrol, no_y_samples, 4) * np.sqrt(vy_MC) + my_MC
            y_samples = my_MC
            Hy = 0
            for t in range(Tcontrol):
                for d in range(4):
                    Hy += entropy(y_samples[t, :, d].reshape([no_y_samples, 1]), k=int(np.sqrt(no_y_samples)))
            print 'trial', trial, 'no_samples', no_y_samples, Hy, np.mean(Hy_fixed_func), np.std(Hy_fixed_func) / np.sqrt(no_func_samples)


def predictive_entropy(params_fname, cval, Tcontrol, M=20, prior=False):
    from mutual_info import entropy
    # load dataset
    data = np.loadtxt('hh_data.txt')
    # use the voltage and potasisum current
    std_data = np.std(data, axis=0)
    data = data / std_data
    y = data[:, :4]
    xc = data[:, [-1]]
    cval = cval / std_data[-1]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    x_control = cval * np.ones([Tcontrol, 1])
    no_panes = 5
    model_aep = aep.SGPSSM(
        y, Dlatent, M, lik='Gaussian', prior_mean=0, prior_var=1000, 
        x_control=xc, gp_emi=True, control_to_emi=True)
    model_aep.load_model(params_fname)
    no_func_samples = 200
    no_y_samples = 300
    Hy_fixed_func = np.zeros(no_func_samples)
    for k in range(no_func_samples):
        if k % 50 == 0:
            print k, no_func_samples
        np.random.seed(k)
        _, my_MC, vy_MC = model_aep.predict_forward_fixed_function(
            Tcontrol, x_control, prop_mode=PROP_MC, no_samples=no_y_samples,
            starting_from_prior=prior)
        # draw y sample
        # y_samples = np.random.randn(Tcontrol, no_y_samples, 4) * np.sqrt(vy_MC) + my_MC
        y_samples = my_MC
        Hy_fixed_func_k = 0
        for t in range(Tcontrol):
            # if t % 20 == 0:
            #     print t, Tcontrol
            for d in range(4):
                Hy_fixed_func_k += entropy(y_samples[t, :, d].reshape([no_y_samples, 1]), k=50)
        Hy_fixed_func[k] = Hy_fixed_func_k

    _, my_MC, vy_MC = model_aep.predict_forward(
        Tcontrol, x_control, prop_mode=PROP_MC, no_samples=no_y_samples)
    # draw y sample
    # y_samples = np.random.randn(Tcontrol, no_y_samples, 4) * np.sqrt(vy_MC) + my_MC
    y_samples = my_MC
    Hy = 0
    for t in range(Tcontrol):
        for d in range(4):
            Hy += entropy(y_samples[t, :, d].reshape([no_y_samples, 1]), k=50)
    return Hy, np.mean(Hy_fixed_func), np.std(Hy_fixed_func) / np.sqrt(no_func_samples)


if __name__ == '__main__':
    M = 30
    alpha = 0.2
    for step in range(5):
        model_fname = 'res/hh_gpssm_MM_M_%d_alpha_%.2f_step_%d.pickle'%(M, alpha, step)
        data_fname = 'res/hh_data_MM_step_%d.txt'%step
        plot_posterior_gp(
            data_fname,
            model_fname,
            'res/hh_gpssm_MM_M_%d_alpha_%.2f_posterior_%d.pdf'%(M, alpha, step),
            M=M)
        plot_prediction_gp_MM(data_fname, model_fname,
                           'res/hh_gpssm_MM_M_%d_alpha_%.2f_prediction_MM_%d.pdf'%(M, alpha, step),
                           M=M)
        plot_prediction_gp_MC(data_fname, model_fname,
                           'res/hh_gpssm_MM_M_%d_alpha_%.2f_prediction_MC_%d.pdf'%(M, alpha, step),
                           M=M)

        # Hy = np.loadtxt('/tmp/hh_gpssm_entropy.txt', delimiter=',')
        # plt.figure()
        # plt.plot(c_vals, Hy[:, 0], '+', color='b')
        # plt.savefig('/tmp/hh_gpssm_Hy.pdf')

        # plt.figure()
        # plt.plot(c_vals, Hy[:, 1], '*', color='r')
        # plt.fill_between(c_vals, Hy[:, 1] + 2*Hy[:, 2], Hy[:, 1] - 2*Hy[:, 2], 
        #     facecolor='r', edgecolor='r', alpha=0.4)
        # plt.savefig('/tmp/hh_gpssm_Hyfixed.pdf')

        # plt.figure()
        # res = Hy[:, 0] - Hy[:, 1]
        # plt.plot(c_vals, res, 'o', color='m')
        # plt.fill_between(c_vals, res + 2*Hy[:, 2], res - 2*Hy[:, 2], 
        #     facecolor='m', edgecolor='m', alpha=0.4)
        # plt.savefig('/tmp/hh_gpssm_Hydiff.pdf')

        # data = np.loadtxt('hh_data.txt')
        # xc = data[:, [-1]]
        # plt.figure()
        # plt.hist(xc, bins=30, range=[-5, 25])
        # plt.savefig('/tmp/hh_gpssm_Hy_xc_hist.pdf')
        #############################

    # K = 10
    # for k in range(K):
    #     print k
    #     np.random.seed(k)
    #     plot_prediction_gp_MM_fixed_function('/tmp/hh_gpssm_M_%d_alpha_%.2f.pickle'%(M, alpha),
    #                        '/tmp/hh_gpssm_M_%d_alpha_%.2f_prediction_MM_fixed_u_%d.pdf'%(M, alpha, k),
    #                        M=M)
    #     np.random.seed(k)
    #     plot_prediction_gp_MC_fixed_function('/tmp/hh_gpssm_M_%d_alpha_%.2f.pickle'%(M, alpha),
    #                        '/tmp/hh_gpssm_M_%d_alpha_%.2f_prediction_MC_fixed_u_%d.pdf'%(M, alpha, k),
    #                        M=M)
    
    # c_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    # Tcontrol = 200
    # for c in c_vals:
    #     print c
    #     plot_prediction_gp_MM_fixed_function_fixed_control(model_fname,
    #                        '/tmp/hh_gpssm_M_%d_alpha_%.2f_prediction_MM_fixed_u_c_%.2f.pdf'%(M, alpha, c),
    #                        c, Tcontrol, M=M)

    #     plot_prediction_gp_MC_fixed_function_fixed_control(model_fname,
    #                        '/tmp/hh_gpssm_M_%d_alpha_%.2f_prediction_MC_fixed_u_c_%.2f.pdf'%(M, alpha, c),
    #                        c, Tcontrol, M=M)


    # #######################
    # from joblib import Parallel, delayed

    # Nc = 100
    # inputs = range(Nc)
    # c_vals = np.linspace(-5, 25, Nc)
    # Tcontrol = 100
    # def compute_entropy(i):
    #     print i, Nc
    #     return predictive_entropy(model_fname, c_vals[i], Tcontrol, M=M)

    # # num_cores = multiprocessing.cpu_count()
    # num_cores = 10
    # results = Parallel(n_jobs=num_cores)(delayed(compute_entropy)(i) for i in inputs)
    # Hy = np.array(results)
    # np.savetxt('/tmp/hh_gpssm_entropy.txt', Hy, delimiter=',', fmt='%.4f')
