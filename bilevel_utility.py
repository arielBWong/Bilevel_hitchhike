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
                               MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, SMD, EI, Surrogate_test

import os
import copy
import multiprocessing as mp
import pygmo as pg
import utilities
from pymop.factory import get_uniform_weights
import EI_krg
from copy import deepcopy
import result_processing


def norm_by_zscore(a, axis=0, ddof=0):
    mns = a.mean(axis=axis, keepdims=True)
    sstd = a.std(axis=axis, ddof=ddof, keepdims=True)
    return (a - mns) / sstd, mns, sstd

def norm_by_exist_zscore(a, mn, sstd):
    return (a - mn)/sstd


def convert_with_zscore(a, mean, std):
    return (a - mean)/std

def reverse_with_zscore(a, mean, std):
    return a * std + mean


def init_xy(number_of_initial_samples, target_problem, seed, **kwargs):

    n_vals = target_problem.n_var
    if len(kwargs) > 0: # bilevel
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
    if len(kwargs) > 0:
        train_y = None
        cons_y = None
    else:
        out = {}
        target_problem._evaluate(train_x, out)
        train_y = out['F']
        train_y = np.atleast_2d(train_y).reshape(number_of_initial_samples, -1)

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


def normalization_with_self(y):
    y = check_array(y)
    min_y = np.min(y, axis=0)
    max_y = np.max(y, axis=0)
    return (y - min_y)/(max_y - min_y)

def search_for_matching_otherlevel_x(x_other, search_iter, n_samples, problem, level, eim, eim_pop, eim_gen,  seed_index, enable_crossvalidation, method_selection):
    train_x, train_y, cons_y = init_xy(n_samples, problem, seed_index,
                                             **{'problem_type': 'bilevel'})

    num_l = train_x.shape[0]
    x_other = np.atleast_2d(x_other)
    xother_expand = np.repeat(x_other, num_l, axis=0)
    if level == 'lower':
        complete_x = np.hstack((xother_expand, train_x))
    else:
        complete_x = np.hstack((train_x, xother_expand))

    complete_y = problem.evaluate(complete_x, return_values_of=["F"])

    for i in range(search_iter):
        new_x = \
            surrogate_search_for_nextx(
                train_x,
                complete_y,
                eim,
                eim_pop,
                eim_gen,
                method_selection,
                enable_crossvalidation)
        # true evaluation to get y
        # add new x-y to training data and repeat

        train_x = np.vstack((train_x, new_x))
        if level == 'lower':
            complete_new_x = np.hstack((x_other, new_x))
        else:
            complete_new_x = np.hstack((new_x, x_other))

        complete_new_y = problem.evaluate(complete_new_x, return_values_of=["F"])
        complete_y = np.vstack((complete_y, complete_new_y))
        # print(np.min(complete_y))

    best_y_index = np.argmin(complete_y)
    best_x = train_x[best_y_index, :]
    np.set_printoptions(precision=2)
    print(best_x)
    np.set_printoptions(precision=2)
    best_y = np.min(complete_y)
    print(np.min(complete_y))

    # conduct local search with true evaluation
    # localsearch_x, est_f = localsearch_on_surrogate(train_x_l, complete_y, best_x, eim)
    localsearch_x, est_f = localsearch_on_trueEvaluation(best_x, level, x_other, problem)

    localsearch_x = np.atleast_2d(localsearch_x)
    if level == 'lower':
        complete_new_x = np.hstack((x_other, localsearch_x))
    else:
        complete_new_x = np.hstack((localsearch_x, x_other))

    localsearch_f = problem.evaluate(complete_new_x, return_values_of=["F"])

    print(localsearch_x)
    print('local search est f %.4f: '% est_f)
    print('local search true f %.4f: '% localsearch_f)
    if localsearch_f < best_y:
        train_x = np.vstack((train_x, localsearch_x))
        complete_y = np.vstack((complete_y, localsearch_f))
        return localsearch_x, localsearch_f, train_x, complete_y

    return np.atleast_2d(best_x), np.atleast_2d(best_y), train_x, complete_y,


def localsearch_on_surrogate(train_x_l, complete_y,  ankor_x, problem):
    # need to update for multiobj!!!
    # conduct local search on the best surrogate
    train_x_l_norm, x_mean, x_std = norm_by_zscore(train_x_l)
    train_y_l_norm, y_mean, y_std = norm_by_zscore(complete_y)

    train_x_l_norm = np.atleast_2d(train_x_l_norm)
    train_y_l_norm = np.atleast_2d(train_y_l_norm)

    krg_l, krg_g_l = cross_val_krg(train_x_l_norm, train_y_l_norm, None, False)

    n_var = train_x_l.shape[1]
    n_constr = 0

    # bounds are zscored
    xbound_l = convert_with_zscore(problem.xl, x_mean, x_std)
    xbound_u = convert_with_zscore(problem.xu, x_mean, x_std)
    ankor_x = np.atleast_2d(ankor_x)
    ankor_x = convert_with_zscore(ankor_x, x_mean, x_std)
    x_bounds = np.vstack((xbound_l, xbound_u)).T.tolist()  # for ea, column direction
    recordFlag = False

    problem_krg = single_krg_optim.single_krg_optim(krg_l[0], n_var, n_constr, 1, xbound_l, xbound_u)

    para = {'add_info': ankor_x}
    x_bounds = np.vstack((xbound_l, xbound_u)).T.tolist()  # for ea, column direction
    recordFlag = False
    pop_test = None
    mut = 0.9
    crossp = 0.1
    popsize = 100
    its = 100

    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x) = \
        optimizer_EI.optimizer(problem_krg,
                               problem_krg.n_obj,
                               problem_krg.n_constr,
                               x_bounds,
                               recordFlag,
                               pop_test,
                               mut,
                               crossp,
                               popsize,
                               its,
                               **para)
    pop_x = reverse_with_zscore(pop_x[0], x_mean, x_std)
    pop_f = reverse_with_zscore(pop_f[0], y_mean, y_std)

    return pop_x, pop_f


def localsearch_on_trueEvaluation(ankor_x, level, other_x, true_problem):
    para = {'add_info': ankor_x,
            'callback': bi_level_compensate_callback,
            'level': level,
            'other_x': other_x,
            }

    x_bounds = np.vstack((true_problem.xl, true_problem.xu)).T.tolist()  # for ea, column direction
    recordFlag = False
    pop_test = None
    mut = 0.9

    crossp = 0.1
    popsize = 20
    its = 20

    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x) = \
        optimizer_EI.optimizer(true_problem,
                               true_problem.n_obj,
                               true_problem.n_constr,
                               x_bounds,
                               recordFlag,
                               pop_test,
                               mut,
                               crossp,
                               popsize,
                               its,
                               **para)
    return pop_x[0], pop_f[0]


def surrogate_search_for_nextx(train_x, train_y, eim, eim_pop, eim_gen, method_selection, enable_crossvalidation):
    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

    train_x_norm = np.atleast_2d(train_x_norm)
    train_y_norm = np.atleast_2d(train_y_norm)

    krg, krg_g = cross_val_krg(train_x_norm, train_y_norm, None, enable_crossvalidation)

    xbound_l = convert_with_zscore(eim.xl, x_mean, x_std)
    xbound_u = convert_with_zscore(eim.xu, x_mean, x_std)
    x_bounds = np.vstack((xbound_l, xbound_u)).T.tolist()  # for ea, column direction

    para = {'train_y': train_y,
            'norm_train_y': train_y_norm,
            'krg': krg,
            'krg_g': krg_g,
            'nadir': None,
            'ideal': None,
            'feasible': np.array([]),
            'ei_method': method_selection}

    recordFlag = False
    pop_x, pop_f = optimizer_EI.optimizer_DE(eim,
                                             eim.n_obj,
                                             eim.n_constr,
                                             x_bounds,
                                             recordFlag,
                                             pop_test=False,
                                             F=0.8,
                                             CR=0.8,
                                             NP=eim_pop,
                                             itermax=eim_gen,
                                             flag=False,
                                             **para)
    pop_x = reverse_with_zscore(pop_x, x_mean, x_std)
    return pop_x


def save_for_count_evaluation(x, train_x, level, x_evaluated):


    if level == 'lower':
        x_expand = np.repeat(np.atleast_2d(x), train_x.shape[0], axis=0)
        x_complete = np.hstack((x_expand, train_x))
    if level == 'upper':
        x_complete = np.hstack((train_x, x))

    x_evaluated = np.vstack((x_evaluated, x_complete))

    return x_evaluated

def problem_test():
    from surrogate_problems import SMD
    problem = SMD.SMD8_f(1, 1, 2)
    xu = []

    # for i in range(2):
    x = np.linspace(-5, 10, 10)
    xu = np.append(xu, x)
    x = np.linspace(-5, 10, 10)
    xu = np.append(xu, x)
    xu = np.reshape(xu, (-1, 2), order='F')

    xl = []
    for i in range(2):
        x = np.linspace(-5, 10, 10)
        xl = np.append(xl, x)
    x = np.linspace(-5, 10, 10)
    xl = np.append(xl, x)
    xl = np.reshape(xl, (-1, 3), order='F')


    x = np.hstack((xu, xl))
    print(x)
    f = problem.evaluate(x, return_values_of=['F'])
    print(f)

def ea_seach_for_matchingx(xu, target_problem_l):
    # how to use ea to conduct search
    bounds = np.vstack((target_problem_l.xl, target_problem_l.xu)).T.tolist()  # for ea, column direction
    paras = {'callback': bi_level_compensate_callback}
    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x) =\
    optimizer_EI.optimizer(target_problem_l,
                           target_problem_l.n_obj,
                           target_problem_l.n_constr,
                           bounds,
                           False,
                           None,
                           0.1,
                           0.9,
                           100,
                           100,
                           **paras)

    np.set_printoptions(precision=2)
    print(pop_x[0])
    print(pop_f[0])
    return pop_x[0], None, None

def bi_level_compensate(level, combine_value, x, activate):
    if not activate:
        return x
    else:
        # x = check_array(x)
        combine_value = check_array(combine_value)
        n = x.shape[0]
        # extend combine values
        combine_value = np.repeat(combine_value, n, axis=0)
        if level =='lower':
            x = np.hstack((combine_value, x))
            return x
        if level == 'upper':
            x = np.hstack((x, combine_value))
            return x


def bi_level_compensate_callback(x, level, compensate):
    x = bi_level_compensate(level, np.atleast_2d(compensate), x, True)
    return x



def save_converge(converge_track, problem_name, method_selection, seed_index):
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_output' + '\\' + problem_name[0:4] + '_' + method_selection
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\converge_' + str(seed_index) + '.csv'
    np.savetxt(saveName, converge_track, delimiter=',')

def save_converge_plot(converge_track, problem_name, method_selection, seed_index):
    import matplotlib.pyplot as plt
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_output' + '\\' + problem_name[0:4] + '_' + method_selection
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\converge_' + str(seed_index) + '.png'
    plt.plot(converge_track)
    plt.title(problem_name + ' seed ' + str(seed_index))
    plt.xlabel('Function evaluation numbers')
    plt.ylabel('F_u')
    # plt.show()
    plt.savefig(saveName)


def save_accuracy(problem_u, problem_l, best_y_u, best_y_l, seed_index, method_selection):
    accuracy_u = np.abs(best_y_u - problem_u.opt)
    accuracy_l = np.abs(best_y_l - problem_l.opt)
    s = [accuracy_u, accuracy_l]
    working_folder = os.getcwd()
    problem = problem_u.name()[0:4]
    result_folder = working_folder + '\\bi_output' + '\\' + problem + '_' + method_selection
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
    np.savetxt(saveName, s, delimiter=',')

# output each problem's output into one single table
def results_process_bestf(BO_target_problems, method_selection):
    import pandas as pd
    n = len(BO_target_problems)
    mean_data = []
    pname_list =[]
    for j in np.arange(0, n, 2):
        target_problem = BO_target_problems[j]
        target_problem = eval(target_problem)
        problem_name = target_problem.name()
        problem_name = problem_name[0:4]
        pname_list.append(problem_name)
        # print(problem_name)
        working_folder = os.getcwd()
        result_folder = working_folder + '\\bi_output' + '\\' + problem_name + '_' + method_selection

        accuracy_data = []
        for seed_index in range(1):
            saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            accuracy_data = np.append(accuracy_data, data)
        accuracy_data = np.atleast_2d(accuracy_data).reshape(-1, 2)
        accuracy_data = np.mean(accuracy_data, axis=0)
        mean_data = np.append(mean_data, accuracy_data)
    mean_data = np.atleast_2d(mean_data).reshape(-1, 2)


    h = pd.DataFrame(mean_data, columns=['ul','ll'], index=pname_list)
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_process'
    saveName = result_folder + '\\accuracy_mean.csv'
    h.to_csv(saveName)




if __name__ == "__main__":

    from EI_krg import expected_improvement

    np.random.seed(0)
    n_samples = 32
    n_iterations = 100
    test_f = Surrogate_test.lower_level_test5()
    eim = \
        EI.EIM(test_f.n_var, n_obj=1, n_constr=test_f.n_constr,
               upper_bound=test_f.xu,
               lower_bound=test_f.xl)

    train_x, train_y, cons_y = init_xy(n_samples, test_f, 0)


    for i in range(n_iterations):
        train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
        train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

        train_x_norm = np.atleast_2d(train_x_norm)
        train_y_norm = np.atleast_2d(train_y_norm)

        krg, krg_g = cross_val_krg(train_x_norm, train_y_norm, None, False)

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



        xbound_l = convert_with_zscore(eim.xl, x_mean, x_std)
        xbound_u = convert_with_zscore(eim.xu, x_mean, x_std)
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
        new_x = reverse_with_zscore(pop_x, x_mean, x_std)
        train_x = np.vstack((train_x, new_x))
        new_y = test_f.evaluate(new_x, return_values_of='F')
        train_y = np.vstack((train_y, new_y))

    best_y_index = np.argmin(train_y)
    best_x = train_x[best_y_index, :]
    np.set_printoptions(precision=2)
    print(best_x)
    np.set_printoptions(precision=2)
    print(np.min(train_y))















