
import numpy as np
import matplotlib.pyplot as plt
import optimizer_EI
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
                               MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, SMD, EI, Surrogate_test,BLTP
from bilevel_utility import localsearch_on_trueEvaluation, bi_level_compensate_callback, search_for_matching_otherlevel_x,\
                            init_xy

import os
import json
import copy

def BLTP4_lviolation(xu, xl1, xl2):

    g1 = -(12 - 4 * xu - 5 * xl1 - 4 * xl2)
    g2 = -(-4 + 4 * xu + 5 * xl1 - 4*xl2)
    g3 = -(4 - 4 * xu + 4 * xl1 - 5 * xl2)
    g4 = -(4 + 4 * xu - 4 * xl1 - 5 * xl2)

    g = [g1, g2, g3, g4]
    g = np.array(g)
    if np.any(g > 0.):
        return False
    else:
        return True


def create_meshgrid(xu):
    xl1 = np.arange(0.85, 0.9, 0.001)
    xl2 = np.arange(0.049, 0.1, 0.001)

    n = len(xl1)
    xl1, xl2 = np.meshgrid(xl1, xl2)

    f = (2 * xl1 - 4) ** 2 + (2 * xl2 - 1) ** 2 + xu * xl1
    orig = copy.deepcopy(f)

    for i in range(n):
        for j in range(n):
            xl1_single = xl1[i, j]
            xl2_single = xl2[i, j]
            if not BLTP4_lviolation(xu, xl1_single, xl2_single):
                f[i, j] = 10

    return xl1, xl2, f, orig


def hybrid_true_search(true_problem, other_x, level):
    para = {
        "callback": bi_level_compensate_callback,
        "level": level,
        "other_x": other_x
    }

    bounds = np.vstack((true_problem.xl, true_problem.xu)).T.tolist()
    # print('before EA search')


    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x) = \
        optimizer_EI.optimizer(true_problem,
                               true_problem.n_obj,
                               true_problem.n_constr,
                               bounds,
                               False,
                               None,
                               0.1,
                               0.9,
                               100,
                               100,
                               **para)
    best_x = pop_x[0]
    best_f = pop_f[0]

    # check feasibility
    best_x = np.atleast_2d(best_x)
    f, c = true_problem.evaluate(np.hstack((other_x, best_x)), return_values_of=['F', 'G'])
    # print('after EA true evaluation: ')
    # print('fl value is: %.4f' % f)
    # print('cons value is ')
    # print(c)
    # if np.any(c > 0):
    # print('EA returns infeasible ')
    # else:
    # print('EA returns feasible solutions')

    best_x, best_f, nfev = localsearch_on_trueEvaluation(best_x, 1000, "lower", other_x, true_problem)

    best_x = np.atleast_2d(best_x)
    f, c = true_problem.evaluate(np.hstack((other_x, best_x)), return_values_of=['F', 'G'])

   #  print('after local search true evaluation: ')
   #  print('fl value is: %.4f' % f)
     # print('cons value is ')
    # print(c)

    c[np.abs(c) < 1e-10] = 0
    if np.any(c > 0):
        # print('local search returns infeasible ')
        flag = False
    else:
        # print('local search returns feasible')
        flag = True

    return best_x, best_f, flag

def mapping_upper_x_and_both_f():
    # this function plot the upper level variable
    # and with this variable, the value of fu and best matching fl

    # two plots are given
    # first is the exghaust search
    # second is the  surrogate search

    # problems_json = 'p/bi_problems'
    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    alg_settings = hyp['alg_settings']
    problem_u = eval(target_problems[0])
    problem_l = eval(target_problems[1])

    number_of_initial_samples = alg_settings['number_of_initial_samples']
    folder = alg_settings['folder']
    lower_interation = alg_settings['lower_interaction']
    stop = alg_settings['stop']
    num_pop = alg_settings['num_pop']
    num_gen = alg_settings['num_gen']

    # set the boundary
    xu_upperbound = problem_u.xu
    xu_lowerbound = problem_u.xl
    xu_range = np.linspace(xu_lowerbound, xu_upperbound, 50)

    eim_l = EI.EIM(problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=problem_l.xu,
                   lower_bound=problem_l.xl)

    # set up a small range for xu
    # xu_range = np.linspace(0, 4, 10000)
    # xu_range = np.linspace(1.8, 1.9, 50)

    # conduct crazy search on lowerlevel
    fu = []
    fu_sur = []
    fl = []
    fl_sur = []

    for xu_each in xu_range:
        xu = np.atleast_2d(xu_each)

        # compare with surrogate returned search
        matching_x, matching_f, n_fev_local, feasible_flag = \
            search_for_matching_otherlevel_x(xu,
                                             lower_interation,
                                             number_of_initial_samples,
                                             problem_l,
                                             'lower',
                                             eim_l,
                                             num_pop,
                                             num_gen,
                                             0,
                                             False,
                                             'eim')
        fl_sur = np.append(fl_sur, matching_f) if feasible_flag else np.append(fl_sur, 10)
        comb = np.hstack((np.atleast_2d(xu), np.atleast_2d(matching_x)))
        f = problem_u.evaluate(comb, return_values_of=['F'])
        fu_sur = np.append(fu_sur, f) if feasible_flag else np.append(fu_sur, 10)

        # try exghaustive search
        best_xl, best_xf, flag = hybrid_true_search(problem_l, xu, 'lower')
        comb = np.hstack((np.atleast_2d(xu), np.atleast_2d(best_xl)))
        f = problem_u.evaluate(comb, return_values_of=['F'])
        if flag:  # lower return feasible
            fu = np.append(fu, f)
            fl = np.append(fl, best_xf)
        else:  # lower return infeasible
            fu = np.append(fu, 10)
            fl = np.append(fl, 10)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)  # row col index
    ax1.scatter(xu_range, fu)
    ax1.scatter(xu_range, fu_sur, marker='+')
    # ax1.scatter(17 / 9, -1.4074)  BLTP5 #
    ax1.scatter(17.4545, -85.0909)   # BLTP9

    ax2 = fig.add_subplot(212)
    ax2.scatter(xu_range, fl)
    # ax2.scatter(17 / 9, 7.6172)  # BLTP5
    ax2.scatter(xu_range, fl_sur, marker='+')
    ax2.scatter(17.4545, -85.0909)  # BlTP9

    plt.show()


def mapping_lowerlevel_landscape():
    # this function uses one upper level variable
    # and then use this xu to
    # construct and plot lower level optimizatio nfunction landscape

    from mpl_toolkits.mplot3d import Axes3D
    # problems_json = 'p/bi_problems'
    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    alg_settings = hyp['alg_settings']

    problem_u = eval(target_problems[0])
    problem_l = eval(target_problems[1])

    # following script is to plot lower level search landscape
    # give a upper level value: currentxu
    # plot lower level search landscape
    xu_i = 50
    # currentxu = xu[xu_i]
    currentxu_one = 1.80783
    # currentxu_one = 17/9

    # prepare for feasible cut
    xl1, xl2, sur, orig = create_meshgrid(currentxu_one)
    sample_num = 200
    xl, _, _ = init_xy(sample_num, problem_l, 0, **{'problem_type': 'bilevel'})

    currentxu = np.atleast_2d(currentxu_one)
    currentxu = np.repeat(currentxu, sample_num, axis=0)

    combo = np.hstack((currentxu, xl))
    fl, fg = problem_l.evaluate(combo, return_values_of=['F', 'G'])

    for i in range(sample_num):
        if np.any(fg[i, :] > 0):
            fl[i, 0] = 10

    xl1 = xl[:, 0]
    xl2 = xl[:, 1]

    # plot with two subplot
    # plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(211)  # row col index
    ax1.set_xlabel('xu')
    ax1.set_ylabel('xu_value')
    ax1.scatter([xu_i], [currentxu_one])
    ax1.set_xlim(0, 100)

    ax2 = fig.add_subplot(2, 1, 2, projection='3d')  # row col index
    ax2.scatter(xl1, xl2, fl)  # , rstride=1, cstride=1,
    # linewidth=0, antialiased=False)

    # ax2.plot_surface(xl1, xl2, orig) #, rstride=1, cstride=1,
    #  linewidth=0, antialiased=False)
    plt.show()


if __name__ == "__main__":
    # mapping_lowerlevel_landscape()
    mapping_upper_x_and_both_f()










