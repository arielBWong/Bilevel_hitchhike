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


def normalization_with_self(y):
    y = check_array(y)
    min_y = np.min(y, axis=0)
    max_y = np.max(y, axis=0)
    return (y - min_y)/(max_y - min_y)

def search_for_matchingx(xu, search_iter, n_samples, problem, level, eim, eim_pop, eim_gen,  seed_index, enable_crossvalidation, method_selection):
    train_x_l, train_y_l, cons_y_l = init_xy(n_samples, problem, seed_index,
                                             **{'problem_type': 'bilevel'})

    num_l = train_x_l.shape[0]
    xu = np.atleast_2d(xu)
    xu_expand = np.repeat(xu, num_l, axis=0)
    complete_x = np.hstack((xu_expand, train_x_l))
    complete_y = problem.evaluate(complete_x, return_values_of=["F"])



    for i in range(search_iter):

        # build model for lower level problem
        norm_complete_y = normalization_with_self(complete_y)
        krg_l, krg_g_l = cross_val_krg(train_x_l, norm_complete_y, cons_y_l, enable_crossvalidation)

        # search for matching lower x

        # search for minimum values of
        para = { 'level': level,
                  'complete': xu,
                  'train_y': complete_y,
                  'norm_train_y': norm_complete_y,
                  'krg': krg_l,
                  'krg_g': krg_g_l,
                  'nadir': None,
                  'ideal': None,
                  'feasible': np.array([]),
                  'ei_method': method_selection}

        x_bounds = np.vstack((problem.xl, problem.xu)).T.tolist()  # for ea, column direction
        recordFlag = False

        pop_x, pop_f = optimizer_EI.optimizer_DE(eim,
                                                 eim.n_obj,
                                                 eim.n_constr,
                                                 x_bounds,
                                                 recordFlag,
                                                 pop_test=None,
                                                 F=0.7,
                                                 CR=0.9,
                                                 NP=eim_pop,
                                                 itermax=eim_gen,
                                                 flag=False,
                                                 **para)
        # evaluate on lower problem
        train_x_l = np.vstack((train_x_l, pop_x))
        complete_new_x = np.hstack((xu, pop_x))
        complete_x = np.vstack((complete_x, complete_new_x))

        complete_new_y = problem.evaluate(complete_new_x, return_values_of=["F"])
        complete_y = np.vstack((complete_y, complete_new_y))
        # print(np.min(complete_y))

    best_y_index = np.argmin(complete_y)
    best_x = train_x_l[best_y_index, :]
    np.set_printoptions(precision=2)
    print(best_x)
    np.set_printoptions(precision=2)
    print(np.min(complete_y))
    print(xu)

    return best_x, train_x_l, complete_y

def surrogate_search_for_optx(train_x, train_y, problem, eim, eim_pop, eim_gen, method_selection, enable_crossvalidation):
    norm_y = normalization_with_self(train_y)
    krg, krg_g = cross_val_krg(train_x, norm_y, None, enable_crossvalidation)

    para = {'level': None,
            'complete': None,
            'train_y': train_y,
            'norm_train_y': norm_y,
            'krg': krg,
            'krg_g': krg_g,
            'nadir': None,
            'ideal': None,
            'feasible': np.array([]),
            'ei_method': method_selection}

    x_bounds = np.vstack((problem.xl, problem.xu)).T.tolist()  # for ea, column direction
    recordFlag = False
    pop_x, pop_f = optimizer_EI.optimizer_DE(eim,
                                             eim.n_obj,
                                             eim.n_constr,
                                             x_bounds,
                                             recordFlag,
                                             pop_test=None,
                                             F=0.8,
                                             CR=0.8,
                                             NP=eim_pop,
                                             itermax=eim_gen,
                                             flag=False,
                                             **para)

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
        x = check_array(x)
        combine_value = check_array(combine_value)
        n = x.shape[0]
        # extend combine values
        combine_value = np.repeat(combine_value, n, axis=0)
        if level=='lower':
            x = np.hstack((combine_value, x))
            return x
        if level == 'upper':
            x = np.hstack((x, combine_value))
            return x


def bi_level_compensate_callback(x):
    x = bi_level_compensate('lower', np.atleast_2d([0]*2), x, True)
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


def save_accuracy(problem_u, problem_l, best_y_u, best_y_l, seed_index,method_selection):
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
        for seed_index in range(11):
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









