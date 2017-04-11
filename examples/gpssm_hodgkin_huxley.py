import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy.stats
import matplotlib.pylab as plt
from .context import aep, ep
np.random.seed(42)
import pdb


def plot_model_no_control(model, plot_title='', name_suffix=''):
    # plot function
    mx, vx = model.get_posterior_x()
    mins = np.min(mx, axis=0) - 0.5
    maxs = np.max(mx, axis=0) + 0.5
    nGrid = 50
    xspaced = np.linspace(mins[0], maxs[0], nGrid)
    yspaced = np.linspace(mins[1], maxs[1], nGrid)
    xx, yy = np.meshgrid(xspaced, yspaced)
    Xplot = np.vstack((xx.flatten(), yy.flatten())).T
    mf, vf = model.predict_f(Xplot)
    fig = plt.figure()
    plt.imshow((mf[:, 0]).reshape(*xx.shape),
               vmin=mf.min(), vmax=mf.max(), origin='lower',
               extent=[mins[0], maxs[0], mins[1], maxs[1]], aspect='auto')
    plt.colorbar()
    plt.contour(
        xx, yy, (mf[:, 0]).reshape(*xx.shape),
        colors='k', linewidths=2, zorder=100)
    zu = model.dyn_layer.zu
    plt.plot(zu[:, 0], zu[:, 1], 'wo', mew=0, ms=4)
    for i in range(mx.shape[0] - 1):
        plt.plot(mx[i:i + 2, 0], mx[i:i + 2, 1],
                 '-bo', ms=3, linewidth=2, zorder=101)
    plt.xlabel(r'$x_{t, 1}$')
    plt.ylabel(r'$x_{t, 2}$')
    plt.xlim([mins[0], maxs[0]])
    plt.ylim([mins[1], maxs[1]])
    plt.title(plot_title)
    plt.savefig('/tmp/hh_gpssm_dim_0' + name_suffix + '.pdf')

    fig = plt.figure()
    plt.imshow((mf[:, 1]).reshape(*xx.shape),
               vmin=mf.min(), vmax=mf.max(), origin='lower',
               extent=[mins[0], maxs[0], mins[1], maxs[1]], aspect='auto')
    plt.colorbar()
    plt.contour(
        xx, yy, (mf[:, 1]).reshape(*xx.shape),
        colors='k', linewidths=2, zorder=100)
    zu = model.dyn_layer.zu
    plt.plot(zu[:, 0], zu[:, 1], 'wo', mew=0, ms=4)
    for i in range(mx.shape[0] - 1):
        plt.plot(mx[i:i + 2, 0], mx[i:i + 2, 1],
                 '-bo', ms=3, linewidth=2, zorder=101)
    plt.xlabel(r'$x_{t, 1}$')
    plt.ylabel(r'$x_{t, 2}$')
    plt.xlim([mins[0], maxs[0]])
    plt.ylim([mins[1], maxs[1]])
    plt.title(plot_title)
    plt.savefig('/tmp/hh_gpssm_dim_1' + name_suffix + '.pdf')


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


def plot_posterior_linear(params_fname, fig_fname, control=False, M=20):
    # load dataset
    data = np.loadtxt('./sandbox/hh_data.txt')
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    if control:
        x_control = xc
        no_panes = 5
    else:
        x_control = None
        no_panes = 4
    model_aep = aep.SGPSSM_Linear(y, Dlatent, M,
                                  lik='Gaussian', prior_mean=0, prior_var=1000, x_control=x_control)
    model_aep.load_model(params_fname)
    my, vy, vyn = model_aep.get_posterior_y()
    vy_diag = np.diagonal(vy, axis1=1, axis2=2)
    vyn_diag = np.diagonal(vyn, axis1=1, axis2=2)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy_diag[:, i]
        vin = vyn_diag[:, i]
        plt.subplot(no_panes, 1, i + 1)
        plt.fill_between(t, mi + 2 * np.sqrt(vi), mi - 2 *
                         np.sqrt(vi), color=cs[i], alpha=0.4)
        plt.plot(t, mi, '-', color=cs[i])
        plt.plot(t, yi, '--', color=cs[i])
        plt.ylabel(labels[i])
        plt.xticks([])
        plt.yticks([])

    if control:
        plt.subplot(no_panes, 1, no_panes)
        plt.plot(t, x_control, '-', color='m')
        plt.ylabel('I')
        plt.yticks([])
    plt.xlabel('t')
    plt.savefig(fig_fname)
    if control:
        plot_model_with_control(model_aep, '', '_linear_with_control')
    else:
        plot_model_no_control(model_aep, '', '_linear_no_control')


def plot_posterior_gp(params_fname, fig_fname, control=False, M=20):
    # load dataset
    data = np.loadtxt('./sandbox/hh_data.txt')
    # use the voltage and potasisum current
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    if control:
        x_control = xc
        no_panes = 5
    else:
        x_control = None
        no_panes = 4
    model_aep = aep.SGPSSM_GP(y, Dlatent, M,
                              lik='Gaussian', prior_mean=0, prior_var=1000, x_control=x_control)
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

    if control:
        plt.subplot(no_panes, 1, no_panes)
        plt.plot(t, x_control, '-', color='m')
        plt.ylabel('I')
        plt.yticks([])
    plt.xlabel('t')

    plt.savefig(fig_fname)

    # if control:
    # 	plot_model_with_control(model_aep, '', '_gp_with_control')
    # else:
    # 	plot_model_no_control(model_aep, '', '_gp_no_control')


def plot_prediction_gp(params_fname, fig_fname, M=20):
    # load dataset
    data = np.loadtxt('./sandbox/hh_data.txt')
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
    model_aep = aep.SGPSSM_GP(y, Dlatent, M,
                              lik='Gaussian', prior_mean=0, prior_var=1000, x_control=x_control)
    model_aep.load_model(params_fname)
    print 'ls ', np.exp(model_aep.dyn_layer.ls)
    my, vy, vyn = model_aep.get_posterior_y()
    mxp, vxp, myp, vyp, vynp = model_aep.predict_forward(T, x_control_test)
    cs = ['k', 'r', 'b', 'g']
    labels = ['V', 'm', 'n', 'h']
    plt.figure()
    t = np.arange(T)
    for i in range(4):
        yi = y[:, i]
        mi = my[:, i]
        vi = vy[:, i]
        vin = vyn[:, i]
        mip = myp[:, i]
        vip = vyp[:, i]
        vinp = vynp[:, i]

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
        plt.yticks([])

    plt.subplot(no_panes, 1, no_panes)
    plt.plot(t, x_control, '-', color='m')
    plt.plot(np.arange(T, 2 * T), x_control_test, '-', color='m')
    plt.axvline(x=T, color='k', linewidth=2)
    plt.ylabel('I')
    plt.yticks([])
    plt.xlabel('t')
    plt.savefig(fig_fname)


def model_linear(params_fname, control=False, M=20, alpha=0.5):
    # load dataset
    data = np.loadtxt('./sandbox/hh_data.txt')
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    R = np.ones(Dobs) * np.log(0.01) / 2
    lsn = np.log(0.01) / 2
    params = {'sn': lsn, 'R': R}

    # create AEP model
    if control:
        x_control = xc
    else:
        x_control = None
    model_aep = aep.SGPSSM_Linear(y, Dlatent, M,
                                  lik='Gaussian', prior_mean=0, prior_var=1000, x_control=x_control)
    hypers = model_aep.init_hypers(y)
    for key in params.keys():
        hypers[key] = params[key]
    model_aep.update_hypers(hypers)
    # optimise
    model_aep.set_fixed_params(['R', 'sn', 'sf'])
    # model_aep.set_fixed_params(['sf'])
    opt_hypers = model_aep.optimise(
        method='L-BFGS-B', alpha=alpha, maxiter=20000, reinit_hypers=False)
    model_aep.save_model(params_fname)


def model_gp(params_fname, control=False, M=20, alpha=0.5):
    # TODO: predict with control
    # load dataset
    data = np.loadtxt('./sandbox/hh_data.txt')
    data = data / np.std(data, axis=0)
    y = data[:, :4]
    xc = data[:, [-1]]
    # init hypers
    Dlatent = 2
    Dobs = y.shape[1]
    T = y.shape[0]
    R = np.log([0.02]) / 2
    lsn = np.log([0.02]) / 2
    params = {'sn': lsn, 'sn_emission': R}

    # create AEP model
    if control:
        x_control = xc
    else:
        x_control = None
    model_aep = aep.SGPSSM_GP(y, Dlatent, M,
                              lik='Gaussian', prior_mean=0, prior_var=1000, x_control=x_control)
    hypers = model_aep.init_hypers(y)
    for key in params.keys():
        hypers[key] = params[key]
    model_aep.update_hypers(hypers)
    # optimise
    # model_aep.set_fixed_params(['sf_emission', 'sf_dynamic'])
    model_aep.set_fixed_params(
        ['sf_emission', 'sf_dynamic', 'sn', 'sn_emission'])
    opt_hypers = model_aep.optimise(
        method='L-BFGS-B', alpha=alpha, maxiter=2000, reinit_hypers=False)
    model_aep.save_model(params_fname)


if __name__ == '__main__':
    M = 30
    alpha = 0.4
    # model_linear('/tmp/hh_gpssm_linear_no_control.pickle', control=False, M=M, alpha=alpha)
    # plot_posterior_linear(
    # 	'/tmp/hh_gpssm_linear_no_control.pickle',
    # 	'/tmp/hh_gpssm_linear_no_control_posterior.pdf',
    # 	control=False, M=M)
    # model_linear('/tmp/hh_gpssm_linear_with_control.pickle', control=True, M=M, alpha=alpha)
    # plot_posterior_linear(
    # 	'/tmp/hh_gpssm_linear_with_control.pickle',
    # 	'/tmp/hh_gpssm_linear_with_control_posterior.pdf',
    # 	control=True, M=M)

    # model_gp('/tmp/hh_gpssm_gp_no_control.pickle', control=False, M=M, alpha=alpha)
    # plot_posterior_gp(
    # 	'/tmp/hh_gpssm_gp_no_control.pickle',
    # 	'/tmp/hh_gpssm_gp_no_control_posterior.pdf',
    # 	control=False, M=M)

    model_gp('/tmp/hh_gpssm_gp_with_control.pickle',
             control=True, M=M, alpha=alpha)
    plot_posterior_gp(
        '/tmp/hh_gpssm_gp_with_control.pickle',
        '/tmp/hh_gpssm_gp_with_control_posterior.pdf',
        control=True, M=M)
    plot_prediction_gp('/tmp/hh_gpssm_gp_with_control.pickle',
                       '/tmp/hh_gpssm_gp_with_control_prediction.pdf',
                       M=M)
