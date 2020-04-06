import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import optimizer_EI
from pymop.factory import get_problem_from_func
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
                  DTLZ1, DTLZ2,\
                  BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
from sklearn.metrics import pairwise_distances
import pyDOE
from cross_val_hyperp import cross_val_krg
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
                               MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, SMD, EI

import os
import copy
import multiprocessing as mp
import pygmo as pg
import utilities
from pymop.factory import get_uniform_weights
import EI_krg
from copy import deepcopy
import result_processing
from bilevel_utility import save_for_count_evaluation, localsearch_on_trueEvaluation, \
    surrogate_search_for_nextx, problem_test, save_converge, save_converge_plot,\
    save_accuracy, results_process_bestf, ea_seach_for_matchingx, search_for_matching_otherlevel_x


def return_current_extreme(train_x, train_y):
    best_index = np.argmin(train_y, axis=0)
    guide_x = train_x[best_index, :]
    return guide_x




def saveNameConstr(problem_name, seed_index, method, run_signature):

    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + problem_name + '_' + run_signature
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)
    # else:
    # os.mkdir(result_folder)
    savename_x = result_folder + '\\best_x_seed_' + str(seed_index) + '_' + method + '.joblib'
    savename_y = result_folder + '\\best_f_seed_' + str(seed_index) + '_' + method +'.joblib'
    savename_FEs = result_folder + '\\FEs_seed_' + str(seed_index) + '_' + method +'.joblib'
    return savename_x, savename_y, savename_FEs


def lexsort_with_certain_row(f_matrix, target_row_index):

    # f_matrix should have the size of n_obj * popsize
    # determine min
    target_row = f_matrix[target_row_index, :].copy()
    f_matrix = np.delete(f_matrix, target_row_index, axis=0)  # delete axis is opposite to normal

    f_min = np.min(f_matrix, axis=1)
    f_min = np.atleast_2d(f_min).reshape(-1, 1)
    # according to np.lexsort, put row with largest min values last row
    f_min_count = np.count_nonzero(f_matrix == f_min, axis=1)
    f_min_accending_index = np.argsort(f_min_count)
    # adjust last_f_pop
    last_f_pop = f_matrix[f_min_accending_index, :]

    # add saved target
    last_f_pop = np.vstack((last_f_pop, target_row))

    # apply np.lexsort (works row direction)
    lexsort_index = np.lexsort(last_f_pop)
    # print(last_f_pop[:, lexsort_index])
    selected_x_index = lexsort_index[0]

    return selected_x_index

def check_krg_ideal_points(krg, n_var, n_constr, n_obj, low, up, guide_x):
    x_krg = []
    f_krg = []

    last_x_pop = []
    last_f_pop = []

    n_krg = len(krg)
    x_pop_size = 100
    x_pop_gen = 100
    add_info = {}

    # identify ideal x and f for each objective
    for k_i, k in enumerate(krg):
        problem = single_krg_optim.single_krg_optim(k, n_var, n_constr, 1, low, up)
        single_bounds = np.vstack((low, up)).T.tolist()

        guide = guide_x[k_i, :]
        #add_info['guide'] = guide

        pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, record = optimizer_EI.optimizer(problem,
                                                                                              nobj=1,
                                                                                              ncon=0,
                                                                                              bounds=single_bounds,
                                                                                              recordFlag=False,
                                                                                              pop_test=None,
                                                                                              mut=0.1,
                                                                                              crossp=0.9,
                                                                                              popsize=x_pop_size,
                                                                                              its=x_pop_gen,
                                                                                              add_info=guide
                                                                                              )
        # save the last population for lexicon sort
        last_x_pop = np.append(last_x_pop, pop_x)
        last_f_pop = np.append(last_f_pop, pop_f)  # var for test

    # long x
    last_x_pop = np.atleast_2d(last_x_pop).reshape(n_obj, -1)

    x_estimate = []
    for i in range(n_obj):
        x_pop = last_x_pop[i, :]
        x_pop = x_pop.reshape(x_pop_size, -1)
        all_f = []
        # all_obj_f under current x pop
        for k in krg:
            f_k, _ = k.predict(x_pop)
            all_f = np.append(all_f, f_k)

        # reorganise all f in obj * popsize shape
        all_f = np.atleast_2d(all_f).reshape(n_obj, -1)
        # select an x according to lexsort
        x_index = lexsort_with_certain_row(all_f, i)

        x_estimate = np.append(x_estimate, x_pop[x_index, :])

    x_estimate = np.atleast_2d(x_estimate).reshape(n_obj, -1)

    return x_estimate

def update_nadir_with_estimate(train_x,  # warning not suitable for more than 3 fs
                               train_y,
                               norm_train_y,
                               cons_y,
                               next_y,
                               problem,
                               x_krg,
                               krg,
                               krg_g,
                               nadir,
                               ideal,
                               enable_crossvalidation,
                               methods_ops,
                               ):

    # add estimated f to train y samples to update ideal and nadir
    n_var = problem.n_var
    n_obj = problem.n_obj
    x1 = np.atleast_2d(x_krg[0]).reshape(-1, n_var)
    x2 = np.atleast_2d(x_krg[1]).reshape(-1, n_var)

    # add new evaluation when next_y is better in any direction compared with
    # current ideal

    if next_y is not None:
        if np.any(next_y < ideal, axis=1):

            # warning: version3, dace is trained on normalized f space
            # for eim version
            train_y_tmp = train_y.copy()
            f1_norm_esti = []
            f2_norm_esti = []
            for k in krg:
                y_norm1, _ = k.predict(x1)
                y_norm2, _ = k.predict(x2)

                f1_norm_esti = np.append(f1_norm_esti, y_norm1)
                f2_norm_esti = np.append(f2_norm_esti, y_norm2)

            # convert back to real scale with ideal and nadir
            f1_norm_esti = np.atleast_2d(f1_norm_esti).reshape(1, -1)
            f2_norm_esti = np.atleast_2d(f2_norm_esti).reshape(1, -1)

            # from this step hv-r3 and eim-3 start to use different processes
            if methods_ops == 'eim_r3':
                # de-normalize back to real range
                f1_esti = f1_norm_esti * (nadir - ideal) + ideal
                f2_esti = f2_norm_esti * (nadir - ideal) + ideal

                # add to existing samples to work out new nadir and ideal
                tmp_sample = np.vstack((train_y_tmp, f1_esti, f2_esti))

                tmp_sample = close_adjustment(tmp_sample)
                nd_front_index = return_nd_front(tmp_sample)
                nd_front = tmp_sample[nd_front_index, :]

                nadir = np.amax(nd_front, axis=0)
                ideal = np.amin(nd_front, axis=0)

                # update krg with one new x/f pair
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

            elif methods_ops == 'hv_r3':
                # hv krg operate on real scale
                # so _norm_ still refer to real scale
                tmp_sample = np.vstack((train_y_tmp, f1_norm_esti, f2_norm_esti))

                tmp_sample = close_adjustment(tmp_sample)
                nd_front_index = return_nd_front(tmp_sample)
                nd_front = tmp_sample[nd_front_index, :]

                nadir = np.amax(nd_front, axis=0)
                ideal = np.amin(nd_front, axis=0)

                # update krg
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None
        else:
            if methods_ops == 'eim_r3':
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
            else:
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None


    return train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal














def update_nadir(train_x,  # warning not suitable for more than 3 fs
                 train_y,
                 norm_train_y,
                 cons_y,
                 next_y,
                 problem,
                 x_krg,
                 krg,
                 krg_g,
                 nadir,
                 ideal,
                 enable_crossvalidation,
                 methods_ops,
                 ):

    '''
    # plot train_y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_y[:, 0], train_y[:, 1], marker='x', c='blue')
    f1 = [nadir[0], nadir[0], ideal[0], ideal[0], nadir[0]]
    f2 = [nadir[1], ideal[1], ideal[1], nadir[1], nadir[1]]
    line = Line2D(f1, f2, c='green')
    ax.add_line(line)
    '''

    # check with new nadir and ideal point
    # update them if they do not meet ideal/nadir requirement
    n_var = problem.n_var
    n_obj = problem.n_obj
    x1 = np.atleast_2d(x_krg[0]).reshape(-1, n_var)
    x2 = np.atleast_2d(x_krg[1]).reshape(-1, n_var)

    # add new evaluation when next_y is better in any direction compared with
    # current ideal
    estimate_flag = False
    if next_y is not None:
        # either add new point and update krg
        # or no adding and only update krg
        if np.any(next_y < ideal, axis=1):
            estimate_flag = True
            # print('new next_y better than ideal')
            # print(next_y)
            # print(ideal)
            out = {}
            problem._evaluate(x1, out)
            y1 = out['F']

            if 'G' in out.keys():
                g1 = out['G']

            problem._evaluate(x2, out)
            y2 = out['F']

            if 'G' in out.keys():
                g2 = out['G']

            # whether there is smaller point than nadir
            train_x = np.vstack((train_x, x1, x2))
            train_y = np.vstack((train_y, y1, y2))
            if 'G' in out.keys():
                cons_y = np.vstack((cons_y, g1, g2))

            # solve the too small distance problem
            train_y = close_adjustment(train_y)
            nd_front_index = return_nd_front(train_y)
            nd_front = train_y[nd_front_index, :]

            nadir = np.amax(nd_front, axis=0)
            ideal = np.amin(nd_front, axis=0)

            # print('ideal update')
            # print(ideal)
            if methods_ops == 'eim_r' or  methods_ops == 'eim_r3':
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

            else:  # hvr/hv_r3
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None
        else:
            if methods_ops == 'eim_r':
                norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
                krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
            else:  # hvr
                krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
                norm_train_y = None



    '''
      
    ax.scatter(y1[:, 0], y1[:, 1], marker='o', c='m')
    ax.scatter(y2[:, 0], y2[:, 1], marker='o', c='red')
   
    # add new line
    f1 = [nadir_new[0], nadir_new[0], ideal_new[0], ideal_new[0], nadir_new[0]]
    f2 = [nadir_new[1], ideal_new[1], ideal_new[1], nadir_new[1], nadir_new[1]]

    line = Line2D(f1, f2, c='red')
    ax.add_line(line)

    ax.scatter(nd_front[:, 0], nd_front[:, 1], c='yellow')
    # ax.scatter(train_y[-1, 0], train_y[-1, 1], marker='D', c='g')
    ax.scatter(y1[:, 0], y1[:, 1], s=200, marker='_', c='m')
    ax.scatter(y2[:, 0], y2[:, 1], s=200, marker='_', c='red')

    up_lim = np.max(np.amax(train_y, axis=0))
    low_lim = np.min(np.amin(train_y, axis=0))
    ax.set(xlim=(low_lim-1, up_lim+1), ylim=(low_lim-1, up_lim+1))
    plt.show()
    '''

    return train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal, estimate_flag



def initNormalization_by_nd(train_y):

    nd_front_index = return_nd_front(train_y)
    nd_front = train_y[nd_front_index, :]
    nadir = np.amax(nd_front, axis=0)
    ideal = np.amin(nd_front, axis=0)

    return nadir, ideal


def init_xy(number_of_initial_samples, target_problem, seed, **kwargs):

    n_vals = target_problem.n_var
    if kwargs is not None: # bilevel
        n_vals = target_problem.n_levelvar
    n_sur_cons = target_problem.n_constr

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples, criterion='maximin')#, iterations=1000)

    xu = np.atleast_2d(target_problem.xu).reshape(1, -1)
    xl = np.atleast_2d(target_problem.xl).reshape(1, -1)

    train_x = xl + (xu - xl) * train_x

    # test
    # lfile = 'sample_x' + str(seed) + '.csv'
    # train_x = np.loadtxt(lfile, delimiter=',')
    if kwargs is not None:
        train_y = None
        cons_y = None
    else:
        out = {}
        target_problem._evaluate(train_x, out)
        train_y = out['F']

        if 'G' in out.keys():
            cons_y = out['G']
            cons_y = np.atleast_2d(cons_y).reshape(-1, n_sur_cons)
        else:
            cons_y = None

    # test
    '''
    lfile = 'sample_x' + str(seed) + '.csv'
    train_x_1 = np.loadtxt(lfile, delimiter=',')
    out = {}
    target_problem._evaluate(train_x_1, out)
    train_y_1 = out['F']

    plt.scatter(train_y[:, 0], train_y[:, 1])
    plt.scatter(train_y_1[:, 0], train_y_1[:, 1])
    plt.legend(['python', 'matlab'])
    plt.show()
    '''

    return train_x, train_y, cons_y


def return_nd_front(train_y):
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
    ndf = list(ndf)

    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    return ndf_extend


def return_hv(nd_front, reference_point, target_problem):
    p_name = target_problem.name()
    n_constr = target_problem.n_constr

    if 'DTLZ' in p_name and int(p_name[-1]) < 5:
        ref_dir = get_uniform_weights(10000, 2)
        true_pf = target_problem.pareto_front(ref_dir)
    elif 'WeldedBeam' in p_name:
        true_pf = target_problem.pareto_front()
    else:
        true_pf = target_problem.pareto_front(n_pareto_points=10000)

    max_by_f = np.amax(true_pf, axis=0)
    min_by_f = np.amin(true_pf, axis=0)

    # normalized to 0-1
    nd_front = (nd_front - min_by_f)/(max_by_f - min_by_f)

    n_obj = nd_front.shape[1]
    n_nd = nd_front.shape[0]

    reference_point_norm = reference_point

    nd_list = []
    for i in range(n_nd):
        if np.all(nd_front[i, :] < reference_point):
            nd_list = np.append(nd_list, nd_front[i, :])
    nd_list = np.atleast_2d(nd_list).reshape(-1, n_obj)

    if len(nd_list) > 0:
        hv = pg.hypervolume(nd_list)
        hv_value = hv.compute(reference_point_norm)
    else:
        hv_value = 0

    return hv_value


def return_igd(target_problem, number_pf_points, nd_front):
    # extract pareto front
    nd_front = check_array(nd_front)
    n_obj = target_problem.n_obj

    # for test
    # nd_front = np.loadtxt('non_dominated_front.csv', delimiter=',')

    if n_obj == 2:
        if 'DTLZ' not in target_problem.name():
            true_pf = target_problem.pareto_front(n_pareto_points=number_pf_points)
        else:
            ref_dir = get_uniform_weights(number_pf_points, 2)
            true_pf = target_problem.pareto_front(ref_dir)

    max_by_f = np.amax(true_pf, axis=0)
    min_by_f = np.amin(true_pf, axis=0)

    # normalized to 0-1
    nd_front = (nd_front - min_by_f) / (max_by_f - min_by_f)

    true_pf = np.atleast_2d(true_pf).reshape(-1, n_obj)
    true_pf = (true_pf - min_by_f)/(max_by_f - min_by_f)

    eu_dist = pairwise_distances(true_pf, nd_front, 'euclidean')
    eu_dist = np.min(eu_dist, axis=1)
    igd = np.mean(eu_dist)
    return igd


def save_hv_igd(train_x, train_y, hv_ref, seed_index, target_problem, method_selection, cons_flag):
    problem_name = target_problem.name()
    n_x = train_x.shape[0]
    nd_front_index = return_nd_front(train_y)
    nd_front = train_y[nd_front_index, :]
    hv = return_hv(nd_front, hv_ref, target_problem)

    # test only
    # compare to matlab code
    if cons_flag == True:
        out = {}
        sample_n = train_x.shape[0]
        target_problem._evaluate(train_x, out)
        cons_y = out['G']
        nd_front = feasible_ndfront(target_problem, train_y, cons_y)
        hv_obj = pg.hypervolume(nd_front)
        hv = hv_obj.compute(hv_ref)


    # for igd, only consider first front

    if cons_flag is not True:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        nd_front = train_y[ndf[0], :]
    igd = return_igd(target_problem, 10000, nd_front)

    save = [hv, igd]
    print('sample size %d, final save hv of current nd_front: %.4f, igd is: %.4f' % (n_x, hv, igd))

    working_folder = os.getcwd()
    result_folder = working_folder + '\\outputs' + '\\' + problem_name + '_' + method_selection
    if not os.path.isdir(result_folder):
        # shutil.rmtree(result_folder)
        # os.mkdir(result_folder)
        os.mkdir(result_folder)
    saveName = result_folder + '\\hv_igd_' + str(seed_index) + '.csv'
    np.savetxt(saveName, save, delimiter=',')




def feasible_ndfront(target_problem, train_y, cons_y_origin):
    # find feasible nd front
    sample_n =  train_y.shape[0]
    n_sur_cons = target_problem.n_constr
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)

    cons_y = copy.copy(cons_y_origin)
    cons_y = np.atleast_2d(cons_y).reshape(-1, n_sur_cons)
    cons_y[cons_y <= 0] = 0
    cons_cv = cons_y.sum(axis=1)
    infeasible = np.nonzero(cons_cv)
    feasible_index = np.setdiff1d(a, infeasible)
    feasible_y = train_y[feasible_index, :]
    if len(feasible_y) > 1:
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(feasible_y)
        ndf = list(ndf)
        nd_front = feasible_y[ndf[0], :]
    else:
        nd_front = feasible_y
    return nd_front




def feasible_check(target_problem, evalparas):
    # extract feasible/feasible_nd

    train_x = evalparas['train_x']
    sample_n = train_x.shape[0]

    n_sur_cons = target_problem.n_constr
    n_obj = target_problem.n_obj
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    out = {}
    target_problem._evaluate(train_x, out)

    if 'G' in out.keys():
        mu_g = out['G']
        mu_g = np.atleast_2d(mu_g).reshape(-1, n_sur_cons)
        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible_index = np.setdiff1d(a, infeasible)

        # mo problem, feasible should be feasible_nd
        if n_obj > 1:
            feasible_y = evalparas['norm_train_y'][feasible_index, :]
            if len(feasible_y) > 1:
                ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(feasible_y)
                ndf = list(ndf)
                evalparas['feasible'] = feasible_y[ndf[0], :]
            else:
                evalparas['feasible'] = feasible_y
        # so problem, feasible is train_y itself
        else:
            feasible_y = evalparas['train_y'][feasible_index, :]
            evalparas['feasible'] = feasible_y

    else:
        evalparas['feasible_norm_nd'] = -1

    return evalparas


def post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature):

    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    # output best archive solutions
    sample_n = train_x.shape[0]
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    out = {}
    target_problem._evaluate(train_x, out)
    if 'G' in out.keys():
        mu_g = out['G']
        mu_g = np.atleast_2d(mu_g).reshape(-1, n_sur_cons)

        mu_g[mu_g <= 0] = 0
        mu_cv = mu_g.sum(axis=1)
        infeasible = np.nonzero(mu_cv)
        feasible = np.setdiff1d(a, infeasible)

        feasible_solutions = train_x[feasible, :]
        feasible_f = train_y[feasible, :]

        n = len(feasible_f)
        # print('number of feasible solutions in total %d solutions is %d ' % (sample_n, n))

        if n > 0:
            best_f = np.argmin(feasible_f, axis=0)
            print('Best solutions encountered so far')
            print(feasible_f[best_f, :])
            best_f_out = feasible_f[best_f, :]
            best_x_out = feasible_solutions[best_f, :]
            print(feasible_solutions[best_f, :])
        else:
            best_f_out = None
            best_x_out = None
            print('No best solutions encountered so far')
    elif n_sur_objs == 1:
        best_f = np.argmin(train_y, axis=0)
        best_f_out = train_y[best_f, :]
        best_x_out = train_x[best_f, :]
    else:
        # print('MO save pareto front from all y')
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        f_pareto = train_y[ndf[0], :]
        best_f_out = f_pareto
        best_x_out = train_x[ndf[0], :]

    savename_x, savename_f, savename_FEs = saveNameConstr(target_problem.name(), seed_index, method_selection, run_signature)

    dump(best_x_out, savename_x)
    dump(best_f_out, savename_f)


def referece_point_check(train_x, train_y, cons_y,  ideal_krg, x_out, target_problem, krg, krg_g, enable_crossvalidation):

    # check whether there is any f that is even better/smaller than ideal
    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

    ideal_true_samples = np.atleast_2d(np.amin(train_y, axis=0))
    compare = np.any(ideal_true_samples < ideal_krg, axis=1)
    # print(ideal_true_samples)
    # print(ideal_krg)
    # print(compare)

    if sum(compare) > 0:
        print('New evaluation')
        # add true evaluation
        for x in x_out:
            x = np.atleast_2d(x).reshape(-1, n_vals)
            out = {}
            target_problem._evaluate(x, out)
            y = out['F']

            train_x = np.vstack((train_x, x))
            train_y = np.vstack((train_y, y))
            if 'G' in out:
                g = np.atleast_2d(out['G']).reshape(-1, n_sur_cons)
                cons_y = np.vstack((cons_y, g))
        # re-conduct krg training
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)
    return krg, krg_g


def normalization_with_nadir_ideal(y, nadir, ideal):
    y = check_array(y)
    return (y - ideal) / (nadir - ideal)


def normalization_with_self(y):
    y = check_array(y)
    min_y = np.min(y, axis=0)
    max_y = np.max(y, axis=0)
    return (y - min_y)/(max_y - min_y)

def normalization_with_nd(y):
    y = check_array(y)
    n_obj = y.shape[1]
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y)
    ndf = list(ndf)
    ndf_size = len(ndf)
    # extract nd for normalization
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = y[ndf_extend, :]

    # normalization boundary
    min_nd_by_feature = np.amin(nd_front, axis=0)
    max_nd_by_feature = np.amax(nd_front, axis=0)

    if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
        print('nd front aligned problem, re-select nd front')
        ndf_index = ndf[0]
        for k in np.arange(1, ndf_size):
            ndf_index = np.append(ndf_index, ndf[k])
            nd_front = y[ndf_index, :]
            min_nd_by_feature = np.amin(nd_front, axis=0)
            max_nd_by_feature = np.amax(nd_front, axis=0)
            if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
                continue
            else:
                break
    norm_y = (y - min_nd_by_feature)/(max_nd_by_feature - min_nd_by_feature)
    return norm_y



def main(seed_index, target_problem, enable_crossvalidation, method_selection, run_signature):

    # this following one line is for work around 1d plot in multiple-processing settings
    mp.freeze_support()
    np.random.seed(seed_index)
    recordFlag = False

    target_problem = eval(target_problem)

    # test variable
    eim_compare = []

    print('Problem %s, seed %d' % (target_problem.name(), seed_index))

    hv_ref = [1.1, 1.1]

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    n_vals = target_problem.n_var

    if n_sur_objs > 2:
        stop = 200
    else:
        stop = 100


    number_of_initial_samples = 11 * n_vals - 1
    n_iter = 300  # stopping criterion set

    if 'WFG' in target_problem.name():
        stop = 250
        number_of_initial_samples = 200


    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem, seed_index)
    # test
    # stop = 400
    # test
    # train_y = np.loadtxt('sample_y.csv', delimiter=',')

    # for evalparas compatibility across differenct algorithms
    # nadir/ideal initialization on nd front no 2d alignment fix
    nadir, ideal = initNormalization_by_nd(train_y)


    # kriging data preparison
    # initialization before infill interation
    if method_selection == 'eim':
        norm_train_y = normalization_with_self(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    elif method_selection == 'eim_nd':
        norm_train_y = normalization_with_nd(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    elif method_selection == 'eim_r':
        norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    elif method_selection == 'eim_r3':
        norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)

    else:
        norm_train_y = None
        krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)

    # always conduct reference search on a initialized samples
    if method_selection == 'hvr' or method_selection == 'hv_r3' or method_selection == 'eim_r' or method_selection == 'eim_r3':
        guide_x = return_current_extreme(train_x, train_y)
        x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu, guide_x)
        next_y = np.atleast_2d([ideal[0]-1, ideal[1]-1])  # force to estimate
        train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal, est_flag = update_nadir(train_x,
                                                                                                    train_y,
                                                                                                    norm_train_y,
                                                                                                    cons_y,
                                                                                                    next_y,  # flag for initialization
                                                                                                    target_problem,
                                                                                                    x_out,
                                                                                                    krg,
                                                                                                    krg_g,
                                                                                                    nadir,
                                                                                                    ideal,
                                                                                                    enable_crossvalidation,
                                                                                                    method_selection,
                                                                                                  )

         # test
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        ndf_size = len(ndf)
        # extract nd for normalization
        if len(ndf[0]) > 1:
            ndf_extend = ndf[0]
        else:
            ndf_extend = np.append(ndf[0], ndf[1])

        nd_front = train_y[ndf_extend, :]
        min_pf_by_feature = ideal
        max_pf_by_feature = nadir
        norm_nd = (nd_front - min_pf_by_feature) / (max_pf_by_feature - min_pf_by_feature)
        hv = pg.hypervolume(norm_nd)
        hv_value = hv.compute([1.1, 1.1])


    # create EI problem
    evalparas = {'train_x':  train_x,
                 'train_y': train_y,
                 'norm_train_y': norm_train_y,
                 'krg': krg,
                 'krg_g': krg_g,
                 'nadir': nadir,
                 'ideal': ideal,
                 'feasible': np.array([]),
                 'ei_method': method_selection}

    # construct ei problems
    ei_problem = get_problem_from_func(acqusition_function,
                                       target_problem.xl,  # row direction
                                       target_problem.xu,
                                       n_var=n_vals,
                                       func_args=evalparas)

    x_bounds = np.vstack((target_problem.xl, target_problem.xu)).T.tolist()  # for ea, column direction

    start_all = time.time()
    # start the searching process
    plot_flag = False
    plt.ion()
    for iteration in range(n_iter):
        print('iteration %d' % iteration)

        # check feasibility in main loop, update evalparas['feasible']
        evalparas = feasible_check(train_x, target_problem, evalparas)

        '''
        if train_x.shape[0] % 5 == 0:
            recordFlag = utilities.intermediate_save(target_problem, method_selection, seed_index, iteration, krg, train_y, nadir, ideal)
        '''

        start = time.time()
        # main loop for finding next x
        candidate_x = np.zeros((1, n_vals))
        candidate_y = []

        num_pop = 50
        num_gen = 50
        if 'ZDT' in target_problem.name():
            num_pop = 200
            num_gen = 200


        for restart in range(4):

            '''
            NSGAII
            pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, record = optimizer_EI.optimizer(ei_problem,
                                                                                                  ei_problem.n_obj,
                                                                                                  ei_problem.n_constr,
                                                                                                  x_bounds,
                                                                                                  recordFlag,
                                                                                                  # pop_test=pop_test,
                                                                                                  pop_test=None,
                                                                                                  mut=0.1,
                                                                                                  crossp=0.9,
                                                                                                  popsize=50,
                                                                                                  its=50,
                                                                                                  **evalparas)

            '''

            # DE
            pop_x, pop_f = optimizer_EI.optimizer_DE(ei_problem,
                                                     ei_problem.n_obj,
                                                     ei_problem.n_constr,
                                                     x_bounds,
                                                     recordFlag,
                                                     # pop_test=pop_test,
                                                     pop_test=None,
                                                     F= 0.8,
                                                     CR= 0.8,
                                                     NP= num_pop,
                                                     itermax=num_gen,
                                                     flag = plot_flag,
                                                     **evalparas)



            candidate_x = np.vstack((candidate_x, pop_x[0, :]))
            candidate_y = np.append(candidate_y, pop_f[0, :])

            '''
            if recordFlag:
                saveName = 'intermediate\\' + target_problem.name() + '_' + method_selection+ '_seed_' + str(seed_index) + 'search_record_iteration_' + str(iteration) + '_restart_' + str(restart) + '.joblib'
                dump(record, saveName)
            '''
        end = time.time()
        lasts = (end - start)



        # print('propose to next x in iteration %d uses %.2f sec' % (iteration, lasts))
        w = np.argwhere(candidate_y == np.min(candidate_y))
        metric_opt = np.min(candidate_y)
        # print('optimization of eim:')
        # eim_compare.append(np.min(candidate_y))
        # print(np.min(candidate_y))

        # propose next_x location
        next_x = candidate_x[w[0]+1, :]
        # test
        # next_x = proposed_x[iteration, :]
        # print(next_x)

        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)




        # generate corresponding f and g
        out = {}
        target_problem._evaluate(next_x, out)
        next_y = out['F']
        # print(next_y)


        '''
        if train_x.shape[0] % 5 == 0:
            saveName  = 'intermediate\\' + target_problem.name() + '_' + method_selection + '_seed_' + str(seed_index) + 'nextF_iteration_' + str(iteration) + '.joblib'
            dump(next_y, saveName)
        '''
        recordFlag = False
        if 'G' in out.keys():
            next_cons_y = out['G']
            next_cons_y = np.atleast_2d(next_cons_y)
        else:
            next_cons_y = None

        # -----------plot -------------
        # plot progress

        plt.clf()
        if 'DTLZ' in target_problem.name() and int(target_problem.name()[-1]) < 5:
            ref_dir = get_uniform_weights(100, 2)
            true_pf = target_problem.pareto_front(ref_dir)
        else:
            true_pf = target_problem.pareto_front(n_pareto_points=100)

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(train_y)
        ndf = list(ndf)
        nd_front = train_y[ndf[0], :]

        f1_pred, _ = krg[0].predict(next_x)
        f2_pred, _ = krg[1].predict(next_x)
        f_pred = np.hstack((f1_pred, f2_pred))

        if method_selection == 'eim_r':
            f_pred = f_pred * (nadir - ideal) + ideal

        if method_selection == 'eim':
            f_min_by_feature = np.amin(train_y, axis=0)
            f_max_by_feature = np.max(train_y, axis=0)
            f_pred = f_pred * (f_max_by_feature - f_min_by_feature) + f_min_by_feature
        if method_selection == 'eim_nd':
            nd_front_index = return_nd_front(train_y)
            nd_front_plot = train_y[nd_front_index, :]
            f_min_by_feature = np.amin(nd_front_plot, axis=0)
            f_max_by_feature = np.max(nd_front_plot, axis=0)
            f_pred = f_pred * (f_max_by_feature - f_min_by_feature) + f_min_by_feature
        if method_selection == 'hv':
            nd_front_index = return_nd_front(train_y)
            nd_front_plot = train_y[nd_front_index, :]
            f_min_by_feature = np.amin(nd_front_plot, axis=0)
            f_max_by_feature = np.max(nd_front_plot, axis=0)
            reference_point = 1.1 * (f_max_by_feature - f_min_by_feature) + f_min_by_feature
        if method_selection == 'hvr' or method_selection == 'hv_r3':
            reference_point = 1.1 * (nadir - ideal) + ideal



        plt.grid(True)
        plt.scatter(true_pf[:, 0], true_pf[:, 1], s=0.2)

        plt.scatter(next_y[:, 0], next_y[:, 1], marker="D", c='red')
        text2 = 'real next y'
        plt.text(next_y[:, 0], next_y[:, 1], text2)

        plt.scatter(train_y[:, 0], train_y[:, 1], marker="o", s=1, c='k')
        plt.scatter(nd_front[:, 0], nd_front[:, 1], marker='o', c='c')
        plt.scatter(f_pred[:, 0], f_pred[:, 1], marker="P")
        text1 = ' predicted next y' + "{:4.2f}".format(metric_opt) + " [{:4.2f}".format(f_pred[0, 0]) + ' ' + "{:4.2f}]".format(f_pred[0, 1])
        plt.text(f_pred[:, 0], f_pred[:, 1], text1)

        if method_selection == 'eim_r' or method_selection == 'eim_r3' or method_selection == 'hv_r3' or method_selection =='hvr':
            plt.scatter(nadir[0], nadir[1], marker='+', c='g')
            plt.text(nadir[0], nadir[1], 'nadir')
            plt.scatter(ideal[0], ideal[1], marker='+', c='g')
            plt.text(ideal[0], ideal[1], 'ideal')
            tt = " [{:4.2f}".format(reference_point[0]) + ' ' + "{:4.2f}]".format(reference_point[1])
            plt.scatter(reference_point[0], reference_point[1], marker='+', c='red')
            plt.text(reference_point[0], reference_point[1]+0.2, tt)

            if iteration == 0:
                plt.scatter(train_y[-1, 0], train_y[-1, 1], marker='x', c='black')
                plt.scatter(train_y[-2, 0], train_y[-2, 1], marker='x', c='black')






        if method_selection == 'eim' or method_selection == 'eim_nd' or method_selection == 'hv':
            plt.scatter(f_min_by_feature[0], f_min_by_feature[1], marker='+', c='g')
            plt.text(f_min_by_feature[0], f_min_by_feature[1], 'ideal')
            plt.scatter(f_max_by_feature[0], f_max_by_feature[1], marker='+', c='g')

            tt = " [{:4.2f}".format(f_max_by_feature[0]) + ' ' + "{:4.2f}]".format(f_max_by_feature[1])
            plt.text(f_max_by_feature[0]-0.2, f_max_by_feature[1]-0.2, tt)
            tt = " [{:4.2f}".format(reference_point[0]) + ' ' + "{:4.2f}]".format(reference_point[1])
            plt.scatter(reference_point[0], reference_point[1], marker='+', c='red')
            plt.text(reference_point[0], reference_point[1], tt)






        # add new proposed data
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        # print('train x  size %d' % train_x.shape[0])

        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))

        #---------
        start = time.time()
        # use extended data to train krging model

        # output hv during the search
        n_x = train_x.shape[0]
        nd_front_index = return_nd_front(train_y)
        nd_front = train_y[nd_front_index, :]
        hv = return_hv(nd_front, hv_ref, target_problem)
        igd = return_igd(target_problem, 10000, nd_front)
        print('iteration: %d, number evaluation: %d, hv of current nd_front: %.4f, igd is: %.4f' % (iteration, n_x, hv, igd))

        #---------plot--------------------------------
        t = 'hv  after adding new point {:6.4f}'.format(hv)
        plt.title(t)




        # kriging  update with newly added x/f
        if method_selection == 'eim':
            norm_train_y = normalization_with_self(train_y)
            krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        elif method_selection == 'eim_nd':
            norm_train_y = normalization_with_nd(train_y)
            krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        # elif method_selection == 'eim_r':
            # norm_train_y = normalization_with_nadir_ideal(train_y, nadir, ideal)
            # krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)
        elif method_selection == 'hv':
            norm_train_y = None
            krg, krg_g = cross_val_krg(train_x, train_y, cons_y, enable_crossvalidation)


        # new evaluation added depending on condition
        if method_selection == 'hvr' or method_selection == 'eim_r':
            if train_x.shape[0] == 70:
                a =0


            guide_x = return_current_extreme(train_x, train_y)
            x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu, guide_x)
            train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal, est_flag = update_nadir(train_x,
                                                                                                        train_y,
                                                                                                        norm_train_y,
                                                                                                        cons_y,
                                                                                                        next_y,
                                                                                                        target_problem,
                                                                                                        x_out,
                                                                                                        krg,
                                                                                                        krg_g,
                                                                                                        nadir,
                                                                                                        ideal,
                                                                                                        enable_crossvalidation,
                                                                                                        method_selection)



            savename = 'visualization\\' + target_problem.name() + '_' + method_selection + '_guide_' + str(
                seed_index) + '_iteration_' + str(train_x.shape[0]) + '.png'
            if est_flag == True:
                plt.scatter(train_y[-1, 0], train_y[-1, 1], marker='x', c='red')
                plt.scatter(train_y[-2, 0], train_y[-2, 1], marker='x', c='red')
            plt.savefig(savename)
            # plt.pause(0.5)

            # -----------plot ends -------------

        # r3 does not add
        if method_selection == 'eim_r3' or method_selection == 'hv_r3':
            guide_x = return_current_extreme(train_x, train_y)
            x_out = check_krg_ideal_points(krg, n_vals, n_sur_cons, n_sur_objs, target_problem.xl, target_problem.xu, guide_x)
            train_x, train_y, norm_train_y, cons_y, krg, krg_g, nadir, ideal = update_nadir_with_estimate(train_x,
                                                                                                          train_y,
                                                                                                          norm_train_y,
                                                                                                          cons_y,
                                                                                                          next_y,
                                                                                                          target_problem,
                                                                                                          x_out,
                                                                                                          krg,
                                                                                                          krg_g,
                                                                                                          nadir,
                                                                                                          ideal,
                                                                                                          enable_crossvalidation,
                                                                                                          method_selection)


            savename = 'visualization\\' + target_problem.name() + '_' + method_selection + '_guide_' + str(
                seed_index) + '_iteration_' + str(train_x.shape[0]) + '.png'
            plt.savefig(savename)
            # plt.pause(0.5)
            # -----------plot ends -------------

        savename = 'visualization\\' + target_problem.name() + '_' + method_selection + '_guide_' + str(
            seed_index) + '_iteration_' + str(train_x.shape[0]) + '.png'
        plt.savefig(savename)

        # -----------plot ends -------------



        lasts = (end - start)
        # print('cross-validation %d uses %.2f sec' % (iteration, lasts))

        # update ea parameters
        evalparas['train_x'] = train_x
        evalparas['train_y'] = train_y
        evalparas['norm_train_y'] = norm_train_y
        evalparas['krg'] = krg
        evalparas['krg_g'] = krg_g
        evalparas['nadir'] = nadir
        evalparas['ideal'] = ideal

        # stopping criteria
        sample_n = train_x.shape[0]
        if sample_n == stop:
            break
        if sample_n > stop:
            vio_more = np.arange(stop, sample_n)
            train_y = np.delete(train_y, vio_more, 0)
            train_x = np.delete(train_x, vio_more, 0)
            break


    plt.ioff()
    end_all = time.time()
    print('overall time %.4f ' % (end_all - start_all))
    save_hv_igd(train_x, train_y, hv_ref, seed_index, target_problem, method_selection)
    post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature)

    # plot
    # result_processing.plot_pareto_vs_ouputs('ZDT3', [seed_index], 'eim', 'eim')
    # savename = 'sample_out_freensga_' + str(seed_index) + '.csv'
    # out = [eim_compare[0], hv, igd]
    # np.savetxt(savename, out, delimiter=',')


def main_mo_c(seed_index, target_problem, enable_crossvalidation, method_selection, run_signature):

    # this following one line is for work around 1d plot in multiple-processing settings
    mp.freeze_support()
    np.random.seed(seed_index)
    plt.ion()
    target_problem = eval(target_problem)
    print('Problem %s, seed %d' % (target_problem.name(), seed_index))

    # collect problem parameters: number of objs, number of constraints
    n_sur_objs = target_problem.n_obj
    n_sur_cons = target_problem.n_constr
    n_vals = target_problem.n_var
    hv_ref = [1.1] * n_sur_objs

    if n_sur_objs > 2:
        stop = 200
    else:
        stop = 100

    # number_of_initial_samples = 11 * n_vals - 1
    number_of_initial_samples = 50
    n_iter = 300  # stopping criterion set

    train_x, train_y, cons_y = init_xy(number_of_initial_samples, target_problem, seed_index)
    nadir, ideal = initNormalization_by_nd(train_y)

    # kriging data preparison
    # initialization before infill interation
    if method_selection == 'eim':
        norm_train_y = normalization_with_self(train_y)
        krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)


    # create infill point search problem
    evalparas = {'train_x':  train_x,
                 'train_y': train_y,
                 'norm_train_y': norm_train_y,
                 'krg': krg,
                 'krg_g': krg_g,
                 'nadir': nadir,
                 'ideal': ideal,
                 'feasible': np.array([]),
                 'ei_method': method_selection}

    # construct ei problems
    ei_problem = get_problem_from_func(acqusition_function,
                                       target_problem.xl,  # row direction
                                       target_problem.xu,
                                       n_var=n_vals,
                                       func_args=evalparas)

    x_bounds = np.vstack((target_problem.xl, target_problem.xu)).T.tolist()  # for ea, column direction

    start_all = time.time()

    # start the infill searching process
    for iteration in range(n_iter):
        print('iteration %d' % iteration)

        # check feasibility in main loop, update evalparas['feasible']
        # feasible is nd-front
        evalparas = feasible_check(target_problem, evalparas)

        start = time.time()
        # main loop for finding next x
        candidate_x = np.zeros((1, n_vals))
        candidate_y = []

        num_pop = 50
        num_gen = 200

        recordFlag = False

        for restart in range(1):

            # DE
            pop_x, pop_f = optimizer_EI.optimizer_DE(ei_problem,
                                                     ei_problem.n_obj,
                                                     ei_problem.n_constr,
                                                     x_bounds,
                                                     recordFlag,
                                                     # pop_test=pop_test,
                                                     pop_test=None,
                                                     F=0.8,
                                                     CR=0.8,
                                                     NP=num_pop,
                                                     itermax=num_gen,
                                                     flag=False,
                                                     **evalparas)



            candidate_x = np.vstack((candidate_x, pop_x[0, :]))
            candidate_y = np.append(candidate_y, pop_f[0, :])

        end = time.time()
        lasts = (end - start)

        # print('propose to next x in iteration %d uses %.2f sec' % (iteration, lasts))
        w = np.argwhere(candidate_y == np.min(candidate_y))
        metric_opt = np.min(candidate_y)

        # propose next_x location
        next_x = candidate_x[w[0]+1, :]

        # dimension re-check
        next_x = np.atleast_2d(next_x).reshape(-1, n_vals)

        # generate corresponding f and g
        out = {}
        target_problem._evaluate(next_x, out)
        next_y = out['F']
        # print(next_y)

        recordFlag = False
        if 'G' in out.keys():
            next_cons_y = out['G']
            next_cons_y = np.atleast_2d(next_cons_y)
        else:
            next_cons_y = None

        #---------plot--------
        # plt.clf()
        # feasible_nd = evalparas['feasible']
        # plt.grid(True)
        # plt.scatter(feasible_nd[:, 0], feasible_nd[:, 1])
        # plt.pause(0.5)
        #---------plot end----



        # add new proposed data x-y-g
        train_x = np.vstack((train_x, next_x))
        train_y = np.vstack((train_y, next_y))
        # print('train x  size %d' % train_x.shape[0])
        if n_sur_cons > 0:
            cons_y = np.vstack((cons_y, next_cons_y))

        #---------
        start = time.time()
        # use extended data to train krging model

        # output hv during the infill search process
        n_x = train_x.shape[0]
        nd_front_index = return_nd_front(train_y)
        nd_front = train_y[nd_front_index, :]
        if n_sur_cons > 0:
            # calculate nd_front among feasible solutions/not normalized
            nd_front = feasible_ndfront(target_problem, train_y, cons_y)

        hv = return_hv(nd_front, hv_ref, target_problem)
        igd = return_igd(target_problem, 10000, nd_front)
        print('iteration: %d, number evaluation: %d, hv of current nd_front: %.4f, igd is: %.4f' % (iteration, n_x, hv, igd))

        # kriging  update with newly added x-f-g
        if method_selection == 'eim':
            norm_train_y = normalization_with_self(train_y)
            krg, krg_g = cross_val_krg(train_x, norm_train_y, cons_y, enable_crossvalidation)


        # update ea parameters
        evalparas['train_x'] = train_x
        evalparas['train_y'] = train_y
        evalparas['norm_train_y'] = norm_train_y
        evalparas['krg'] = krg
        evalparas['krg_g'] = krg_g
        evalparas['nadir'] = nadir
        evalparas['ideal'] = ideal

        # stopping criteria
        sample_n = train_x.shape[0]
        if sample_n == stop:
            break
        if sample_n > stop:
            vio_more = np.arange(stop, sample_n)
            train_y = np.delete(train_y, vio_more, 0)
            train_x = np.delete(train_x, vio_more, 0)
            break

    plt.ioff()
    end_all = time.time()
    print('overall time %.4f ' % (end_all - start_all))


    # save_hv_igd(train_x, train_y, hv_ref, seed_index, target_problem, method_selection,True)
    save_hv_igd(train_x, train_y, [100, 0.08], seed_index, target_problem, method_selection, True)
    post_process(train_x, train_y, cons_y, target_problem, seed_index, method_selection, run_signature)




def main_bi_mo(seed_index, target_problem, enable_crossvalidation, method_selection, run_signature):
    # this following one line is for work around 1d plot in multiple-processing settings
    mp.freeze_support()
    np.random.seed(seed_index)
    plt.ion()
    target_problem_u = eval(target_problem[0])
    target_problem_l = eval(target_problem[1])
    print('Problem %s, seed %d' % (target_problem_u.name(), seed_index))
    print('Problem %s, seed %d' % (target_problem_l.name(), seed_index))

    # number_of_initial_samples = 11 * n_vals - 1
    number_of_initial_samples = 20
    n_iter = 80  # stopping criterion set
    converge_track = []
    lower_interation = 30
    stop = 80
    start = time.time()

    # init upper level variables, bilevel only init xu no evaluation is done
    train_x_u, train_y_u, cons_y_u = init_xy(number_of_initial_samples, target_problem_u, seed_index, **{'problem_type':'bilevel'})

    # test purpose for validation only
    # train_x_u = np.atleast_2d([0] * target_problem_u.n_levelvar)
    # train_x_u = np.repeat(train_x_u, number_of_initial_samples, axis=0)

    eim_l = EI.EIM(target_problem_l.n_levelvar, n_obj=1, n_constr=target_problem_l.n_constr,
                   upper_bound=target_problem_l.xu,
                   lower_bound=target_problem_l.xl)

    # for each upper level variable, search for its corresponding lower variable for compensation
    num_u = train_x_u.shape[0]
    num_pop = 50
    num_gen = 50

    # circumstantial varaible for saving
    x_evaluated_u = np.atleast_2d(np.zeros((1, target_problem_u.n_var)))
    x_evaluated_l = np.atleast_2d(np.zeros((1, target_problem_l.n_levelvar)))
    y_evaluated_u = np.atleast_2d(np.zeros((1, target_problem_u.n_obj)))
    y_evaluated_l = np.atleast_2d(np.zeros((1, target_problem_l.n_obj)))

    # record matching f
    matching_xl = []
    matching_fl = []
    for xu in train_x_u:
        matching_x, matching_f, _, _ = \
            search_for_matching_otherlevel_x(xu,
                                             lower_interation,
                                             number_of_initial_samples,
                                             target_problem_l,
                                             'lower',
                                             eim_l,
                                             num_pop,
                                             num_gen,
                                             seed_index,
                                             enable_crossvalidation,
                                             method_selection)


        # matching_x, train_x_l, train_y_l = ea_seach_for_matchingx(xu, target_problem_l)
        # test stop there
        # no more saving lower level each evaluations
        # x_evaluated_l = save_for_count_evaluation(xu, train_x_l, 'lower', x_evaluated_l)
        # y_evaluated_l = np.vstack((y_evaluated_l, train_y_l))

        matching_fl = np.append(matching_fl, matching_f)
        matching_xl = np.append(matching_xl, matching_x)
    matching_xl = np.atleast_2d(matching_xl).reshape(-1, target_problem_l.n_levelvar)
    matching_fl = np.atleast_2d(matching_fl).reshape(-1, target_problem_l.n_obj)

    # true evaluation
    complete_x_u = np.hstack((train_x_u, matching_xl))
    complete_y_u = target_problem_u.evaluate(complete_x_u, return_values_of=["F"])

    # count true evaluations
    x_evaluated_u = np.vstack((x_evaluated_u, complete_x_u))
    y_evaluated_u = np.vstack((y_evaluated_u, complete_y_u))
    # before entering evaluation, adjusting save, delete first row
    x_evaluated_u = np.delete(x_evaluated_u, obj=0, axis=0)
    y_evaluated_u = np.delete(y_evaluated_u, obj=0, axis=0)

    x_evaluated_l = np.vstack((x_evaluated_l, matching_xl))
    y_evaluated_l = np.vstack((y_evaluated_l, matching_fl))

    x_evaluated_l = np.delete(x_evaluated_l, obj=0, axis=0)
    y_evaluated_l = np.delete(y_evaluated_l, obj=0, axis=0)

    # search for opt xu
    eim_u = EI.EIM(target_problem_u.n_levelvar, n_obj=1, n_constr=target_problem_u.n_constr,
                   upper_bound=target_problem_u.xu,
                   lower_bound=target_problem_u.xl)
    searched_xu = \
        surrogate_search_for_nextx(train_x_u,
                                   complete_y_u,
                                   eim_u,
                                   num_pop,
                                   num_gen,
                                   method_selection,
                                   enable_crossvalidation)

    # find lower level problem complete for this new pop_x
    for i in range(n_iter):
        print('iteration %d' % i)
        matching_xl, matching_fl, _, _ = \
            search_for_matching_otherlevel_x(searched_xu,
                                             lower_interation,
                                             number_of_initial_samples,
                                             target_problem_l,
                                             'lower',
                                             eim_l,
                                             num_pop,
                                             num_gen,
                                             seed_index,
                                             enable_crossvalidation,
                                             method_selection)

        # combine matching xl to xu for true evaluation
        matching_xl = np.atleast_2d(matching_xl)
        new_complete_xu = np.hstack((searched_xu, matching_xl))
        new_complete_yu = target_problem_u.evaluate(new_complete_xu, return_values_of=["F"])
        print('iteration %d, yu true evaluated: %f' % (i, new_complete_yu))

        # save training data compete lower and upper
        x_evaluated_u = np.vstack((x_evaluated_u, new_complete_xu))
        y_evaluated_u = np.vstack((y_evaluated_u, new_complete_yu))
        x_evaluated_l = np.vstack((x_evaluated_l, matching_xl))
        y_evaluated_l = np.vstack((y_evaluated_l, matching_fl))

        # adding new xu yu to training
        train_x_u = np.vstack((train_x_u, searched_xu))
        complete_y_u = np.vstack((complete_y_u, new_complete_yu))

        n = x_evaluated_u.shape[0]
        if n > stop:
            break

        # if evaluation limit is not reached, search for next xu
        searched_xu = \
            surrogate_search_for_nextx(train_x_u,
                                       complete_y_u,
                                       eim_u,
                                       num_pop,
                                       num_gen,
                                       method_selection,
                                       enable_crossvalidation)


        # record convergence
        converge_track.append(np.min(y_evaluated_u))
        bu = np.min(y_evaluated_u)
        print('iteration %d, yu true evaluated/best so far: %.4f/%.4f ' % (i, new_complete_yu, bu))


    # conduct a local search based on fl
    min_fu_index = np.argmin(y_evaluated_u)
    best_xu_sofar = x_evaluated_u[min_fu_index, 0:target_problem_u.n_levelvar]
    matching_xl = x_evaluated_l[min_fu_index, :]

    localsearch_xl, localsearch_fl = localsearch_on_trueEvaluation(matching_xl, 'lower', best_xu_sofar, target_problem_l)

    new_local_xu, new_local_fu, _, _ = \
        search_for_matching_otherlevel_x(localsearch_xl,
                                         30,
                                         20,
                                         target_problem_u,
                                         'upper',
                                         eim_u,
                                         num_pop,
                                         num_gen,
                                         seed_index,
                                         enable_crossvalidation,
                                         method_selection)

    new_complete_x = np.hstack((np.atleast_2d(new_local_xu), np.atleast_2d(localsearch_xl)))
    new_fl = target_problem_l.evaluate(new_complete_x, return_values_of=["F"])
    new_fu = target_problem_u.evaluate(new_complete_x, return_values_of=["F"])
    y_evaluated_u = np.vstack((y_evaluated_u, new_fu))
    y_evaluated_l = np.vstack((y_evaluated_l, new_fl))
    x_evaluated_u = np.vstack((x_evaluated_u, new_complete_x))
    x_evaluated_l = np.vstack((x_evaluated_l, np.atleast_2d(matching_xl)))
    converge_track.append(new_fu)


    end = time.time()
    duration = (end - start) / 60
    print('overall time used: %0.4f mins' % duration)


    save_accuracy(target_problem_u, target_problem_l, new_fu[0,0], new_fl[0,0], seed_index, method_selection)
    # save_converge(converge_track, target_problem_u.name(), method_selection, seed_index)
    save_converge_plot(converge_track, target_problem_u.name(), method_selection, seed_index)


    return None



if __name__ == "__main__":




    MO_target_problems = [
                          # 'BNH()',
                          #  'TNK()',
                          # 'WeldedBeam()'
                          # 'ZDT1(n_var=6)',
                          # 'ZDT2(n_var=6)',
                          # 'ZDT3(n_var=6)',
                          #'WFG.WFG_1(n_var=2, n_obj=2, K=1)',
                          # 'WFG.WFG_2(n_var=6, n_obj=2, K=4)',
                          # 'WFG.WFG_3(n_var=6, n_obj=2, K=4)',
                           # 'WFG.WFG_4(n_var=6, n_obj=2, K=4)',
                         #'WFG.WFG_5(n_var=6, n_obj=2, K=4)',
                          #'WFG.WFG_6(n_var=6, n_obj=2, K=4)',
                          #'WFG.WFG_7(n_var=6, n_obj=2, K=4)',
                          #'WFG.WFG_8(n_var=6, n_obj=2, K=4)',
                          #'WFG.WFG_9(n_var=6, n_obj=2, K=4)',
                           # 'DTLZ1(n_var=6, n_obj=2)',
                          #  'DTLZ2(n_var=6, n_obj=2)',
                         # 'DTLZs.DTLZ5(n_var=6, n_obj=2)',
                      #'DTLZs.DTLZ7(n_var=6, n_obj=2)',
                          # 'iDTLZ.IDTLZ1(n_var=6, n_obj=2)',
                          # 'iDTLZ.IDTLZ2(n_var=6, n_obj=2)',
                        ]
    BO_target_problems = ['SMD.SMD1_F(1,1,2)',
                          'SMD.SMD1_f(1,1,2)',
                          'SMD.SMD2_F(1,1,2)',
                          'SMD.SMD2_f(1,1,2)',
                          'SMD.SMD3_F(1,1,2)',
                          'SMD.SMD3_f(1,1,2)',
                          'SMD.SMD4_F(1,1,2)',
                          'SMD.SMD4_f(1,1,2)',
                          'SMD.SMD5_F(1,1,2)',
                          'SMD.SMD5_f(1,1,2)',
                          'SMD.SMD6_F(1,1,0,2)',
                          'SMD.SMD6_f(1,1,0,2)',
                          'SMD.SMD7_F(1,1,2)',
                          'SMD.SMD7_f(1,1,2)',
                          'SMD.SMD8_F(1,1,2)',
                          'SMD.SMD8_f(1,1,2)',
                          ]



    run_sig = ['eim']  # 'hv_r3']  #'eim_nd', 'eim', 'eim_r', 'eim_r3']
    methods_ops = ['eim']   # 'hv_r3']  # 'eim_nd', 'eim', 'eim_r', 'eim_r3']  #, 'hv', 'eim_r', 'hvr',  'eim','eim_nd' ]
    args = []
    for seed in range(1, 12):
        for target_problem in MO_target_problems:
            for method in methods_ops:
                args.append((seed, target_problem, False, method, method))

    n = len(BO_target_problems)
    for seed in range(0, 1):
        for j in np.arange(0, n, 2):
            for method in methods_ops:
                target_problem = BO_target_problems[j: j+2]
                args.append((seed, target_problem, False, method, method))
    # main_mo_c(1, MO_target_problems[0], False, 'eim', 'eim')
    i = 2
    main_bi_mo(0, BO_target_problems[i:i+2], False, 'eim', 'eim')
    # problem_test()

    # num_workers = 6
    # pool = mp.Pool(processes=num_workers)
    # pool.starmap(main_bi_mo, ([arg for arg in args]))

    results_process_bestf(BO_target_problems, 'eim')

    ''' 
    target_problems = [branin.new_branin_5(),
                       Gomez3.Gomez3(),
                       Mystery.Mystery(),
                       Reverse_Mystery.ReverseMystery(),
                       SHCBc.SHCBc(),
                       Haupt_schewefel.Haupt_schewefel(),
                       HS100.HS100(),
                       GPc.GPc()]
    '''









