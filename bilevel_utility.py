import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import optimizer_EI
from pymop.factory import get_problem_from_func
from pymop import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6, \
                  DTLZ1, DTLZ2,\
                  BNH, Carside, Kursawe, OSY, Truss2D, WeldedBeam, TNK
from EI_krg import acqusition_function, close_adjustment
from sklearn.utils.validation import check_array
import scipy
from sklearn.metrics import pairwise_distances
import pyDOE
from cross_val_hyperp import cross_val_krg
from joblib import dump, load
import time
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
                               MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, SMD, EI, Surrogate_test

import os
import json
from copy import deepcopy



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

    if problem.n_constr > 0:
        print("constraint settings")
        complete_y, complete_c = problem.evaluate(complete_x, return_values_of=["F", "G"])
    else:
        complete_y = problem.evaluate(complete_x, return_values_of=["F"])
        complete_c = None

    for i in range(search_iter):
        new_x = \
            surrogate_search_for_nextx(
                train_x,
                complete_y,
                complete_c,
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

        if problem.n_constr > 0:
            print('constraints setting')
            complete_new_y, complete_new_c = problem.evaluate(complete_new_x, return_values_of=["F", "G"])
            complete_c = np.vstack((complete_c, complete_new_c))
        else:
            complete_new_y = problem.evaluate(complete_new_x, return_values_of=["F"])

        complete_y = np.vstack((complete_y, complete_new_y))

        # print(np.min(complete_y))

    if problem.n_constr > 0:
        print('constr process')
        complete_y, train_x = return_feasible(complete_c, complete_y, train_x)

    best_y_index = np.argmin(complete_y)
    best_x = train_x[best_y_index, :]
    best_x = np.atleast_2d(best_x)

    best_y = np.min(complete_y)

    # conduct local search with true evaluation
    # localsearch_x, est_f = localsearch_on_surrogate(train_x_l, complete_y, best_x, eim)
    localsearch_x, localsearch_f, n_fev = localsearch_on_trueEvaluation(best_x, 250, level, x_other, problem)

    n_fev = n_fev + search_iter + n_samples

    if localsearch_f < best_y:
        train_x = np.vstack((train_x, localsearch_x))
        complete_y = np.vstack((complete_y, localsearch_f))
        return localsearch_x, localsearch_f, n_fev, train_x, complete_y

    return np.atleast_2d(best_x), np.atleast_2d(best_y), n_fev, train_x, complete_y,


def localsearch_for_matching_otherlevel_x(x_other, max_eval, search_level, problem, seed_index):
    # random init for x
    dimensions = problem.n_levelvar
    start_x = np.random.rand(1, dimensions)
    start_x = problem.xl + start_x * np.fabs(problem.xu - problem.xl)
    x_other = np.atleast_2d(x_other)
    start_x = np.atleast_2d(start_x)
    localsearch_x, f, nfev = localsearch_on_trueEvaluation(start_x, max_eval, search_level, x_other, problem)
    localsearch_x = np.atleast_2d(localsearch_x)

    return localsearch_x, f, nfev, None, None


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

def hybridsearch_on_trueEvaluation(ankor_x, level, other_x, true_problem):

    para = {
        "add_info": ankor_x,
        "callback": bi_level_compensate_callback,
        "level": level,
        "other_x": other_x
    }
    bounds = np.vstack((true_problem.xl, true_problem.xu)).T.tolist()

    # call for global evaluation
    best_x, best_f = \
        optimizer_EI.optimizer_DE(true_problem,
                                  true_problem.n_obj,
                                  true_problem.n_constr,
                                  bounds,
                                  False,
                                  None,
                                  0.7,
                                  0.9,
                                  20,
                                  50,
                                  False,
                                  **para)

    best_x, best_f, nfev = localsearch_on_trueEvaluation(best_x, 250, "lower", other_x, true_problem)
    nfev = nfev + 20*50
    return best_x, best_f, nfev


def localsearch_on_trueEvaluation(ankor_x, max_eval, level, other_x, true_problem):
    ankor_x = np.atleast_2d(ankor_x)
    ankor_x = check_array(ankor_x)

    def obj_func(x):
        x = np.atleast_2d(x)
        if level == 'lower':
            x = np.hstack((other_x, x))
        else:
            x = np.hstack((x, other_x))
        return true_problem.evaluate(x,  return_values_of=["F"])

    def cons_func(x):
        x = np.atleast_2d(x)
        if level == 'lower':
            x = np.hstack((other_x, x))
        else:
            x = np.hstack((x, other_x))
        return true_problem.evaluate(x, return_values_of=["G"])

    bounds = scipy.optimize.Bounds(lb=true_problem.xl, ub=true_problem.xu)
    if true_problem.n_constr > 0:
        optimization_res = scipy.optimize.minimize(
            obj_func, ankor_x, method="SLSQP", options={'maxiter': max_eval},
            constraint={'type': 'ineq', 'fun': cons_func}, jac=False, bounds=bounds)
    else:
        optimization_res = scipy.optimize.minimize(
            obj_func, ankor_x, method="SLSQP", options={'maxiter': max_eval}, jac=False,
            bounds=bounds)

    print('number of function evaluations: %d '% optimization_res.nfev)

    x, f, num_fev = optimization_res.x, optimization_res.fun, optimization_res.nfev
    return x, f, num_fev

def return_feasible(solutions_c, solutions_y, solution_x):
    sample_n = solutions_c.shape[0]
    a = np.linspace(0, sample_n - 1, sample_n, dtype=int)
    solutions_c[solutions_c <= 0] = 0
    solutions_c_violation = solutions_c.sum(axis=1)
    infeasible = np.nonzero(solutions_c_violation)
    feasible = np.setdiff1d(a, infeasible)

    return solutions_y[feasible, :], solution_x[feasible, :]


def surrogate_search_for_nextx(train_x, train_y, train_c, eim, eim_pop, eim_gen, method_selection, enable_crossvalidation):
    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

    train_x_norm = np.atleast_2d(train_x_norm)
    train_y_norm = np.atleast_2d(train_y_norm)

    if train_c is not None:
        train_c_norm, c_mean, c_std = norm_by_zscore(train_c)
        train_c_norm = np.atleast_2d(train_c_norm)
    else:
        train_c_norm = None

    krg, krg_g = cross_val_krg(train_x_norm, train_y_norm, train_c_norm, enable_crossvalidation)

    xbound_l = convert_with_zscore(eim.xl, x_mean, x_std)
    xbound_u = convert_with_zscore(eim.xu, x_mean, x_std)
    x_bounds = np.vstack((xbound_l, xbound_u)).T.tolist()  # for ea, column direction

    # construct feasible for para
    if train_c is not None:
        feasible_norm, _ = return_feasible(train_c, train_y_norm, train_x_norm)
    else:
        feasible_norm = np.array([])


    para = {'train_y': train_y,
            'norm_train_y': train_y_norm,
            'krg': krg,
            'krg_g': krg_g,
            'nadir': None,
            'ideal': None,
            'feasible': np.array(feasible_norm),
            'ei_method': method_selection}

    recordFlag = False
    pop_x, pop_f = optimizer_EI.optimizer_DE(eim,
                                             eim.n_obj,
                                             eim.n_constr,
                                             x_bounds,
                                             recordFlag,
                                             pop_test=False,
                                             F=0.7,
                                             CR=0.9,
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

def save_converge_plot(converge_track, problem_name, method_selection, seed_index, folder):
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_output' + '\\' + problem_name[0:4] + '_' + method_selection
    result_folder = working_folder + '\\bi_ego_output' + '\\' + problem_name[0:4] + '_' + method_selection
    result_folder = working_folder + '\\' + folder + '\\' + problem_name[0:4] + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\converge_' + str(seed_index) + '.png'
    if os.path.exists(saveName):
        os.remove(saveName)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(converge_track)
    t = problem_name + ' seed ' + str(seed_index)
    ax.set_title(t)
    ax.set_xlabel('Function evaluation numbers')
    ax.set_ylabel('F_u')
    # plt.show()
    plt.savefig(saveName)

def save_accuracy(problem_u, problem_l, best_y_u, best_y_l, seed_index, method_selection, folder):
    accuracy_u = np.abs(best_y_u - problem_u.opt)
    accuracy_l = np.abs(best_y_l - problem_l.opt)
    s = [accuracy_u, accuracy_l]
    working_folder = os.getcwd()
    problem = problem_u.name()[0:4]
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
    np.savetxt(saveName, s, delimiter=',')

# output each problem's output into one single table
def results_process_bestf(BO_target_problems, method_selection):
    import pandas as pd
    n = len(BO_target_problems)
    mean_data = []
    median_data = []
    pname_list =[]
    for j in np.arange(6, 8, 2):
        target_problem = BO_target_problems[j]
        target_problem = eval(target_problem)
        problem_name = target_problem.name()
        problem_name = problem_name[0:4]
        pname_list.append(problem_name)
        # print(problem_name)
        working_folder = os.getcwd()
        result_folder = working_folder + '\\bi_output' + '\\' + problem_name + '_' + method_selection
        # result_folder = working_folder + '\\bi_ego_output' + '\\' + problem_name + '_' + method_selection
        # result_folder = working_folder + '\\bi_local_output' + '\\' + problem_name + '_' + method_selection

        accuracy_data = []
        for seed_index in range(29):
            # saveName = result_folder + '\\accuracy_before_evaluation' + str(seed_index) + '.csv'
            saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            accuracy_data = np.append(accuracy_data, data)
        accuracy_data = np.atleast_2d(accuracy_data).reshape(-1, 2)
        accuracy_mean = np.mean(accuracy_data, axis=0)
        accuracy_median = np.median(accuracy_data, axis=0)
        mean_data = np.append(mean_data, accuracy_mean)
        median_data = np.append(median_data, accuracy_median)

    mean_data = np.atleast_2d(mean_data).reshape(-1, 2)
    median_data = np.atleast_2d(median_data).reshape(-1, 2)

    h = pd.DataFrame(mean_data, columns=['ul', 'll'], index=pname_list)
    h2 = pd.DataFrame(median_data, columns=['ul', 'll'], index=pname_list)
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_process'
    saveName = result_folder + '\\ego_accuracy_mean_4.csv'
    saveName2 = result_folder + '\\ego_accuracy_median_4.csv'
    h.to_csv(saveName)
    h2.to_csv(saveName2)

def combine_fev(BO_target_problems, method_selection):
    import pandas as pd
    n = len(BO_target_problems)

    median_data = []

    folders = ['bi_output', 'bi_local_output']
    mean_cross_strategies = []
    for folder in folders:
        pname_list = []
        mean_smds = []
        for j in np.arange(0, n, 2):
            target_problem = BO_target_problems[j]
            target_problem = eval(target_problem)
            problem_name = target_problem.name()
            problem_name = problem_name[0:4]
            pname_list.append(problem_name)
            # print(problem_name)
            working_folder = os.getcwd()
            # result_folder = working_folder + '\\bi_output' + '\\' + problem_name + '_' + method_selection
            result_folder = working_folder + '\\' + folder + '\\' + problem_name + '_' + method_selection
            # result_folder = working_folder + '\\bi_local_output' + '\\' + problem_name + '_' + method_selection

            n_fev = []
            for seed_index in range(11):
                # saveName = result_folder + '\\accuracy_before_evaluation' + str(seed_index) + '.csv'
                saveName = result_folder + '\\ll_nfev' + str(seed_index) + '.csv'
                data = np.loadtxt(saveName, delimiter=',')
                n_fev = np.append(n_fev, data)
            mean_nfev = np.median(n_fev)
            mean_smds = np.append(mean_smds, mean_nfev)

        mean_cross_strategies = np.append(mean_cross_strategies, mean_smds)


    mean_cross_strategies = np.atleast_2d(mean_cross_strategies).reshape(-1, 2)

    h = pd.DataFrame(mean_cross_strategies, columns=['Combined', 'Local only'], index=pname_list)
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_process'
    saveName = result_folder + '\\compare_fev_median.csv'
    h.to_csv(saveName)


def results_process_before_after(BO_target_problems, method_selection, alg_folder, accuracy_name):
    import pandas as pd
    n = len(BO_target_problems)
    mean_data = []
    median_data = []
    pname_list =[]
    for j in np.arange(0, n, 2):
        target_problem = BO_target_problems[j]
        target_problem = eval(target_problem)
        problem_name = target_problem.name()
        problem_name = problem_name[0:4]
        pname_list.append(problem_name)
        # print(problem_name)
        working_folder = os.getcwd()
        result_folder = working_folder + '\\' + alg_folder + '\\' + problem_name + '_' + method_selection

        accuracy_data = []
        for seed_index in range(11):
            saveName = result_folder + '\\' + accuracy_name + '_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            accuracy_data = np.append(accuracy_data, data)
        accuracy_data = np.atleast_2d(accuracy_data).reshape(-1, 2)
        accuracy_mean = np.mean(accuracy_data, axis=0)
        accuracy_median = np.median(accuracy_data, axis=0)
        mean_data = np.append(mean_data, accuracy_mean)
        median_data = np.append(median_data, accuracy_median)

    mean_data = np.atleast_2d(mean_data).reshape(-1, 2)
    median_data = np.atleast_2d(median_data).reshape(-1, 2)

    return median_data

def process_before_after(BO_target_problems, method_selection, alg_folder):

    n = len(BO_target_problems)
    fu_before = []
    fu_after = []
    fl_before = []
    fl_after = []
    pname_list =[]
    for j in np.arange(0, n, 2):
        target_problem = BO_target_problems[j]
        target_problem = eval(target_problem)
        problem_name = target_problem.name()
        problem_name = problem_name[0:4]
        pname_list.append(problem_name)
        # print(problem_name)
        working_folder = os.getcwd()
        result_folder = working_folder + '\\' + alg_folder + '\\' + problem_name + '_' + method_selection

        accuracy_data = []
        for seed_index in range(11):
            saveName = result_folder + '\\accuracy_before_reevaluation_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            accuracy_data = np.append(accuracy_data, data[0])

        # just decide which seed to use
        xu_median_index = np.argsort(accuracy_data)[5]

        accuracy_median = accuracy_data[xu_median_index]
        fu_before = np.append(fu_before, accuracy_median)

        saveName = result_folder + '\\accuracy_' + str(xu_median_index) + '.csv'
        data = np.loadtxt(saveName, delimiter=',')
        fu_after = np.append(fu_after, data[0])
        fl_after = np.append(fl_after, data[1])

        saveName = result_folder + '\\accuracy_before_reevaluation_' + str(xu_median_index) + '.csv'
        data = np.loadtxt(saveName, delimiter=',')
        fl_before = np.append(fl_before, data[1])


    return np.atleast_2d(fu_before).reshape(-1, 1), \
           np.atleast_2d(fu_after).reshape(-1, 1),\
           np.atleast_2d(fl_before).reshape(-1, 1),\
           np.atleast_2d(fl_after).reshape(-1, 1)


def outer_process(BO_target_problems, method_selection):
    import pandas as pd

    alg_folders = ['bi_output', 'bi_local_output'] # 'bi_output', , 'bi_ego_output',
    accuracy_names = ['accuracy', 'accuracy_before_reevaluation']
    # upper - 0/lower - 1
    level = 1
    save_u = np.atleast_2d(np.zeros((int(len(BO_target_problems)/2), 1)))
    save_l = np.atleast_2d(np.zeros((int(len(BO_target_problems) / 2), 1)))

    for folder in alg_folders:
        before_u, after_u, before_l, after_l = \
            process_before_after(BO_target_problems, method_selection, folder) # 0-upper, 1-lower
        save = np.hstack((before_u, after_u, before_l, after_l ))

        h = pd.DataFrame(save,
                         columns=[# 'Combine before', 'Combine after',
                                 'upper level before', 'upper level after',
                                 'lower level before', 'lower level after'],
                         index=['SMD1', 'SMD2', 'SMD3', 'SMD4',
                                'SMD5', 'SMD6', 'SMD7', 'SMD8'])
        working_folder = os.getcwd()
        result_folder = working_folder + '\\bi_process'
        savename = result_folder + '\\before_after_compare_' + folder + '.csv'
        h.to_csv(savename)


def save_before_reevaluation(problem_u, problem_l, xu, xl, fu, fl, seed_index,
                         method_selection, folder):
    accuracy_u = np.abs(fu - problem_u.opt)
    accuracy_l = np.abs(fl - problem_l.opt)
    s = [accuracy_u, accuracy_l]
    working_folder = os.getcwd()
    problem = problem_u.name()[0:4]
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    saveName = result_folder + '\\accuracy_before_reevaluation_' + str(seed_index) + '.csv'
    np.savetxt(saveName, s, delimiter=',')


def save_function_evaluation(nfev, problem, seed_index, method_selection, folder):
    working_folder = os.getcwd()
    problem = problem.name()[0:4]
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    saveName = result_folder + '\\ll_nfev' + str(seed_index) + '.csv'
    np.savetxt(saveName, [nfev], delimiter=',')

def multiple_algorithm_results_combine():
    import pandas as pd

    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_process'
    combined = result_folder + '\\accuracy_median.csv'
    ego = result_folder + '\\ego_accuracy_median.csv'
    local = result_folder + '\\local_accuracy_median.csv'

    combined_data = pd.read_csv(combined)
    ego_data = pd.read_csv(ego)
    local_data = pd.read_csv(local)
    n_problem = len(combined_data.index)
    # print(combined_data['ll'])
    # print(np.array(combined_data['ll']))

    # create combined matrix
    level = 'ul'
    compare_u = []
    compare_u = np.append(compare_u, np.array(combined_data[level]))
    compare_u = np.append(compare_u, np.array(ego_data[level]))
    compare_u = np.append(compare_u, np.array(local_data[level]))
    compare_u = np.atleast_2d(compare_u).reshape(n_problem, -1, order='F')

    # re-write into file
    h = pd.DataFrame(compare_u, columns=['combined', 'ego_only', 'local_search_only'], index=combined_data.index)
    saveName = result_folder + '\\' + level + '_compare.csv'
    h.to_csv(saveName)


if __name__ == "__main__":
    # problem
    problems_json = 'p/bi_problems'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    results_process_bestf(target_problems, 'eim')








    '''
    score=[1,2,3,4,5]

    with open("file.json", 'w') as f:
        # indent=2 is not needed but makes the file more
        # human-readable for more complicated data
 
        json.dump(score, f, indent=2)
    
    with open("file.json", 'r') as f:
        score = json.load(f)

    print(score)
    '''














