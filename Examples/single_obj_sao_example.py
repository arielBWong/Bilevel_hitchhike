import numpy as np
import matplotlib.pyplot as plt
import optimizer_EI
from cross_val_hyperp import cross_val_krg
from surrogate_problems import EI, Surrogate_test
import bilevel_utility
from EI_krg import expected_improvement

if __name__ == "__main__":

    n_samples = 5
    n_iterations = 10
    test_f = Surrogate_test.f()
    eim = \
        EI.EIM(test_f.n_var, n_obj=1, n_constr=test_f.n_constr,
                   upper_bound=test_f.xu,
                   lower_bound=test_f.xl)

    train_x, train_y, cons_y = bilevel_utility.init_xy(n_samples, test_f, 0)

    plt.ion()
    fig = plt.figure()
    test_x = np.linspace(test_f.xl, test_f.xu, 1000)
    test_y = test_f.evaluate(test_x, return_values_of='F')

    for i in range(n_iterations):
        train_x_norm, x_mean, x_std = bilevel_utility.norm_by_zscore(train_x)
        train_y_norm, y_mean, y_std = bilevel_utility.norm_by_zscore(train_y)

        train_x_norm = np.atleast_2d(train_x_norm)
        train_y_norm = np.atleast_2d(train_y_norm)

        krg, krg_g = cross_val_krg(train_x_norm, train_y_norm, None, False)

        # plot result
        test_x_norm = bilevel_utility.norm_by_exist_zscore(test_x, x_mean, x_std)

        pred_y_norm, pred_y_sig_norm = krg[0].predict(test_x_norm)
        pred_y = bilevel_utility.reverse_with_zscore(pred_y_norm, y_mean, y_std)

        # use test_x_norm to plot eim

        #------------------------------
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('design variable')
        ax1.set_ylabel('f and predicted f value')
        ax1.plot(test_x, test_y, 'r')
        ax1.plot(test_x, pred_y, 'b')
        # ax1.plot(test_x, pred_y + pred_y_sig_norm, 'y')
        # ax1.plot(test_x, pred_y - pred_y_sig_norm, 'y')
        ax1.fill_between(test_x.ravel(), (pred_y + pred_y_sig_norm).ravel(), (pred_y - pred_y_sig_norm).ravel(), alpha=0.5)
        ax1.scatter(train_x, train_y)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('expected improvement')  # we already handled the x-label with ax1
        # ------------------------------


        para = {'level': None,
                'complete': None,
                'train_y': train_y,
                'norm_train_y': train_y_norm,
                'krg': krg,
                'krg_g': krg_g,
                'nadir': None,
                'ideal': None,
                'feasible': np.array([]),
                'ei_method': 'eim'}

        # -------------------------------
        ei = expected_improvement(test_x_norm, **para)
        ax2.plot(test_x, ei, '--g')
        # -------------------------------

        xbound_l = bilevel_utility.convert_with_zscore(eim.xl, x_mean, x_std)
        xbound_u = bilevel_utility.convert_with_zscore(eim.xu, x_mean, x_std)
        x_bounds = np.vstack((xbound_l, xbound_u)).T.tolist()  # for ea, column direction
        recordFlag = False


        pop_x, pop_f = optimizer_EI.optimizer_DE(eim,
                                                 eim.n_obj,
                                                 eim.n_constr,
                                                 x_bounds,
                                                 recordFlag,
                                                 pop_test=None,
                                                 F=0.7,
                                                 CR=0.9,
                                                 NP=50,
                                                 itermax=50,
                                                 flag=False,
                                                 **para)
        # evaluate on lower problem
        new_x = bilevel_utility.reverse_with_zscore(pop_x, x_mean, x_std)
        train_x = np.vstack((train_x, new_x))
        new_y = test_f.evaluate(new_x, return_values_of='F')
        train_y = np.vstack((train_y, new_y))

        # -------------------------------
        ax1.scatter(new_x.ravel(), new_y.ravel())
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.legend(['True function', 'Estimation', 'Sigma', 'Training data', 'Proposed point'], loc='upper left')
        ax2.legend(['EI'], loc='upper right')
        plt.pause(2)
        fig.clf()
        # -------------------------------

    plt.ioff()

