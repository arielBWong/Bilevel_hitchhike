import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import optimizer_EI
from sklearn.utils.validation import check_array
import scipy
import pyDOE
from cross_val_hyperp import cross_val_krg
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
                               MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, SMD, EI, Surrogate_test,BLTP

import os
import json
import copy
import joblib



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

def feasibility_adjustment(part_x, combine_x, combine_y, combine_c, feasibility):
    feasibility = np.array(feasibility)
    infeasible_index = np.argwhere(feasibility == False)
    infeasible_index = infeasible_index.ravel()
    part_x = np.delete(part_x, infeasible_index, axis=0)
    combine_x = np.delete(combine_x, infeasible_index, axis=0)
    combine_y = np.delete(combine_y, infeasible_index, axis=0)
    combine_c = np.delete(combine_c, infeasible_index, axis=0)
    return part_x, combine_x, combine_y, combine_c

# infeasible values are set high
def feasibility_adjustment_2(part_x, combine_x, combine_y, combine_c, feasibility):
    feasibility = np.array(feasibility)
    infeasible_index = np.argwhere(feasibility == False)
    infeasible_index = infeasible_index.ravel()
    combine_y[infeasible_index, :] = 1e6
    return part_x, combine_x, combine_y, combine_c


def search_for_matching_otherlevel_x(x_other, search_iter, n_samples, problem, level, eim, eim_pop, eim_gen,  seed_index, enable_crossvalidation, method_selection, **kwargs):
    train_x, train_y, cons_y = init_xy(n_samples, problem, seed_index,
                                             **{'problem_type': 'bilevel'})

    num_l = train_x.shape[0]
    # print(train_x[0, :])
    x_other = np.atleast_2d(x_other)
    xother_expand = np.repeat(x_other, num_l, axis=0)
    if level == 'lower':
        complete_x = np.hstack((xother_expand, train_x))
    else:
        complete_x = np.hstack((train_x, xother_expand))

    if problem.n_constr > 0:
        # print("constraint settings")
        complete_y, complete_c = problem.evaluate(complete_x, return_values_of=["F", "G"])

    else:
        complete_y = problem.evaluate(complete_x, return_values_of=["F"])
        complete_c = None

    for i in range(search_iter):
        new_x, _, _ = \
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

        # include evaluated cons and f
        if problem.n_constr > 0:
            # print('constraints setting')
            complete_new_y, complete_new_c = problem.evaluate(complete_new_x, return_values_of=["F", "G"])
            complete_c = np.vstack((complete_c, complete_new_c))
        else:
            complete_new_y = problem.evaluate(complete_new_x, return_values_of=["F"])
        complete_y = np.vstack((complete_y, complete_new_y))

    # after ego, decide where to start local search
    if problem.n_constr > 0:
        # print('constr process')
        complete_y_feasible, train_x_feasible = return_feasible(complete_c, complete_y, train_x)
        if len(complete_y_feasible) > 0:
            complete_y = complete_y_feasible
            train_x = train_x_feasible
            best_y_index = np.argmin(complete_y)
            best_x = train_x[best_y_index, :]
            best_x = np.atleast_2d(best_x)
            best_y = np.min(complete_y)
        else:
            print("no feasible found while ego on searching matching x")
            best_x, best_y = nofeasible_select(complete_c, complete_y, train_x)
            print("before local search, closest best y: %.4f" % best_y)
    else:
        best_y_index = np.argmin(complete_y)
        best_x = train_x[best_y_index, :]
        best_x = np.atleast_2d(best_x)
        best_y = np.min(complete_y)

    # conduct local search with true evaluation
    localsearch_x, localsearch_f, n_fev = localsearch_on_trueEvaluation(best_x, 250, level, x_other, problem)
    n_fev = n_fev + search_iter + n_samples

    print('local search %s level, before %.4f, after %.4f' % (level, best_y, localsearch_f))
    print('local search found lx')
    print(localsearch_x)

    if problem.n_constr > 0:
        if len(complete_y_feasible) > 0:
            print('feasible exist after surrogate for upper xu %0.4f' % x_other[0, 0])
        else:
            print('no feasible after surrogate for upper xu %0.4f' % x_other[0, 0])



    # decide which x is returned
    if problem.n_constr > 0:
        # check feasibility of local search
        if level == 'lower':
            c = problem.evaluate(np.hstack((x_other, np.atleast_2d(localsearch_x))), return_values_of=["G"])
        else:
            c = problem.evaluate(np.hstack((np.atleast_2d(localsearch_x), x_other)), return_values_of=["G"])
        print('local search return constraint:')
        print(c)
        c[np.abs(c) < 1e-10] = 0

        # surrogate found feasible, consider local returns feasible
        # decide on smaller f value
        if len(complete_y_feasible) > 0:
            if np.any(c > 0):   # surrogate has feasible, local search has no
                print('surrogate returns feasible, but local search not, return surrogate results')
                print('surrogate constraint output is:')
                sc = problem.evaluate(np.hstack((x_other, np.atleast_2d(best_x))), return_values_of=["G"])
                print(sc)
                return_tuple = (np.atleast_2d(best_x), np.atleast_2d(best_y), n_fev, True)
            else:
                print('both surrogate and local search returns feasible, return smaller one')
                feasible_flag = True
                return_tuple = (localsearch_x, localsearch_f, n_fev, feasible_flag) \
                    if localsearch_f < best_y \
                    else (np.atleast_2d(best_x), np.atleast_2d(best_y), n_fev, feasible_flag)
        # surrogate did not find any feasible solutions
        # decide (1) local search has feasible return local results
        # (2) local search has no feasible, flag infeasible, return whatever xl fl
        else:
            # localsearch also finds no feasible
            # only  flag matters, as this solution on upper level will be ignored or set 1e6
            if np.any(c > 0):
                print('local search also has no feasible solution, return false flag')
                feasible_flag = False
                return_tuple = (localsearch_x, localsearch_f, n_fev, feasible_flag)
            else:  # local search found feasible while surrogate did not
                print('local search found feasible while surrogate did not')
                return_tuple = (localsearch_x, localsearch_f, n_fev, True)
    # no-constraint problem process
    else:
        return_tuple = (localsearch_x, localsearch_f, n_fev, True) \
            if localsearch_f < best_y \
            else (np.atleast_2d(best_x), np.atleast_2d(best_y), n_fev, True)

    return return_tuple[0], return_tuple[1], return_tuple[2], return_tuple[3]





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
    print('before EA search')

    f, g = true_problem.evaluate(np.atleast_2d(np.hstack((other_x, ankor_x))), return_values_of=['F', 'G'])
    print('fl value is: %.4f' % f)
    print('cons value is ')
    print(g)

    # call for global evaluation
    '''
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
    '''
    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g, (record_f, record_x) = \
        optimizer_EI.optimizer(true_problem,
                               true_problem.n_obj,
                               true_problem.n_constr,
                               bounds,
                               False,
                               None,
                               0.1,
                               0.9,
                               20,
                               50,
                               **para)
    best_x = pop_x[0]
    best_f = pop_f[0]


    # check feasibility
    best_x = np.atleast_2d(best_x)
    f, c = true_problem.evaluate(np.hstack((other_x, best_x)), return_values_of=['F', 'G'])
    print('after EA true evaluation: ')
    print('fl value is: %.4f' % f)
    print('cons value is ')
    print(c)
    # if np.any(c > 0):
        # print('EA returns infeasible ')
    # else:
        # print('EA returns feasible solutions')

    best_x, best_f, nfev = localsearch_on_trueEvaluation(best_x, 250, "lower", other_x, true_problem)
    nfev = nfev + 20*50

    best_x = np.atleast_2d(best_x)
    f, c = true_problem.evaluate(np.hstack((other_x, best_x)),  return_values_of=['F', 'G'])
    print('after local search true evaluation: ')
    print('fl value is: %.4f' % f)
    print('cons value is ')
    print(c)
    # c[np.abs(c) < 1e-10] = 0
    # if np.any(c > 0):
        # print('local search returns infeasible ')
    # else:
        # print('local search returns feasible')




    return best_x, best_f, nfev


def localsearch_on_trueEvaluation(ankor_x, max_eval, level, other_x, true_problem):
    ankor_x = np.atleast_2d(ankor_x)
    other_x = np.atleast_2d(other_x)
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
        constr = true_problem.evaluate(x, return_values_of=["G"])
        constr = constr * -1
        return constr.ravel()

    bounds = scipy.optimize.Bounds(lb=true_problem.xl, ub=true_problem.xu)
    if true_problem.n_constr > 0:
        optimization_res = scipy.optimize.minimize(
            obj_func, ankor_x, method="SLSQP", options={'maxiter': max_eval},
            constraints={'type': 'ineq', 'fun': cons_func}, jac=False, bounds=bounds)
    else:
        optimization_res = scipy.optimize.minimize(
            obj_func, ankor_x, method="SLSQP", options={'maxiter': max_eval}, jac=False,
            bounds=bounds)

    print('number of function evaluations: %d '% optimization_res.nfev)

    x, f, num_fev = optimization_res.x, optimization_res.fun, optimization_res.nfev
    return x, f, num_fev

def nofeasible_select(constr_c, train_y, train_x):

    # infeasible selection
    # come to this method means no feasibles
    # so select smallest infeasible
    constr_c = np.sum(constr_c, axis=1)
    feas_closest = np.argmin(constr_c)
    return np.atleast_2d(train_x[feas_closest, :]), np.atleast_2d(train_y[feas_closest, :])

def return_feasible(solutions_c_orig, solutions_y, solution_x):
    solutions_c = copy.deepcopy(solutions_c_orig)
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

    # construct feasible for dict: para
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
    return pop_x, krg, krg_g


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
    result_folder = working_folder + '\\bi_output' + '\\' + problem_name[0:-2] + '_' + method_selection
    result_folder = working_folder + '\\bi_ego_output' + '\\' + problem_name[0:-2] + '_' + method_selection
    result_folder = working_folder + '\\' + folder + '\\' + problem_name[0:-2] + '_' + method_selection

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
    problem = problem_u.name()[0:-2]
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
    np.savetxt(saveName, s, delimiter=',')

def saveKRGmodel(krg, krg_g, folder, problem_u, seed_index):
    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_krgmodels'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    krgmodel_save = result_folder + '\\krg_' + str(seed_index) + '.joblib'
    joblib.dump(krg, krgmodel_save)

    krgmodel_save = result_folder + '\\krg_g_' + str(seed_index) + '.joblib'
    joblib.dump(krg_g, krgmodel_save)

def rebuild_surrogate_and_plot():
    print('not yet need to re-run')


def saveEGOtraining(complete_xu, complete_yu, folder, problem_u):

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    egodata_save = result_folder + '\\sampled_data_x.csv'
    np.savetxt(egodata_save, complete_xu, delimiter=',')

    egodata_save = result_folder + '\\sampled_data_y.csv'
    np.savetxt(egodata_save, complete_yu, delimiter=',')

def EGO_rebuildplot(problem_u, folder):
    # load data
    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    name_x = result_folder + '\\sampled_data_x.csv'
    name_y = result_folder + '\\sampled_data_y.csv'

    complete_x = np.loadtxt(name_x, delimiter=',')
    complete_y = np.loadtxt(name_y, delimiter=',')

    complete_x = np.atleast_2d(complete_x)
    train_y = np.atleast_2d(complete_y).reshape(-1, 1)
    train_x = np.atleast_2d(complete_x[:, 0]).reshape(-1, 1)

    test_x, _ ,_ = init_xy(1000, problem_u, 0, **{'problem_type': 'bilevel'})

    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

    train_x_norm = np.atleast_2d(train_x_norm)
    train_y_norm = np.atleast_2d(train_y_norm)

    krg, krg_g = cross_val_krg(train_x_norm, train_y_norm, None, False)

    # plot result
    test_x_norm = norm_by_exist_zscore(test_x, x_mean, x_std)

    pred_y_norm, pred_y_sig_norm = krg[0].predict(test_x_norm)
    pred_y = reverse_with_zscore(pred_y_norm, y_mean, y_std)

    # ------------------------------
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('design variable')
    ax1.set_ylabel('predicted f value')
    # ax1.set_ylim(-2, 10)
    ax1.scatter(test_x, pred_y)
    plt.show()


def save_feasibility(problem_u, problem_l, up_feas, low_feas, seed_index, method_selection, folder):
    s = [up_feas, low_feas]
    working_folder = os.getcwd()
    problem = problem_u.name()[0:-2]
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    saveName = result_folder + '\\feasibility_' + str(seed_index) + '.csv'
    np.savetxt(saveName, s, delimiter=',')



def results_process_bestf(BO_target_problems, method_selection, seedmax, folder):
    import pandas as pd
    n = len(BO_target_problems)
    median_across_problems = []
    feas_across_problems = []
    median_seed_accrossProblems = []
    pname_list = []
    for j in np.arange(0, n, 2):
        target_problem = BO_target_problems[j]
        target_problem = eval(target_problem)
        problem_name = target_problem.name()
        problem_name = problem_name[0:-2]
        pname_list.append(problem_name)
        # print(problem_name)
        working_folder = os.getcwd()
        result_folder = working_folder + '\\' + folder + '\\' + problem_name + '_' + method_selection

        accuracy_data = []
        for seed_index in range(seedmax):
            # saveName = result_folder + '\\accuracy_before_evaluation' + str(seed_index) + '.csv'
            saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            accuracy_data = np.append(accuracy_data, data)

        accuracy_data = np.atleast_2d(accuracy_data).reshape(-1, 2)

        # select median with ul and corresponding ll
        ul_accuracy = np.array(accuracy_data[:, 0])
        seed_median  = np.argsort(ul_accuracy)[int(seedmax/2)]
        # accuracy_median is for one value
        accuracy_median = accuracy_data[seed_median, :]

        # median_problems save accuracy for across problem
        median_across_problems = np.append(median_across_problems, accuracy_median)
        median_seed_accrossProblems = np.append(median_seed_accrossProblems, seed_median)

        # all problems save feasibility
        # even uncontraint problems
        feasi_data = []  # feasibility across seeds
        for seed_index in range(seedmax):
            saveName = result_folder + '\\feasibility_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            feasi_data = np.append(feasi_data, data)
        feasi_data = np.atleast_2d(feasi_data).reshape(-1, 2)  # 2 -- as upper and lower feasibility
        # pick up feasility corresponding to accuracy median
        feasi_selected = feasi_data[seed_median, :]
        feas_across_problems = np.append(feas_across_problems, feasi_selected)

    # re-arrange problem accuracy accross problems
    median_across_problems = np.atleast_2d(median_across_problems).reshape(-1, 2)
    median_seed_accrossProblems = np.atleast_2d(median_seed_accrossProblems).reshape(-1, 1)


    # compatible with constraints

    feas_across_problems = np.atleast_2d(feas_across_problems).reshape(-1, 2)
    acc_fesi_output = np.hstack((median_across_problems, feas_across_problems, median_seed_accrossProblems))
    h1 = pd.DataFrame(acc_fesi_output, columns=['ul', 'll','ufeasi', 'lfeasi', 'seed'], index=pname_list)
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_process'
    saveName1 = result_folder + '\\ego_accuracy_feasi_median.csv'
    h1.to_csv(saveName1)
    return None

    # h2 = pd.DataFrame(median_across_problems, columns=['ul', 'll'], index=pname_list)
    # working_folder = os.getcwd()
    # result_folder = working_folder + '\\bi_process'
    # saveName2 = result_folder + '\\ego_accuracy_median.csv'
    # h2.to_csv(saveName2)






def combine_fev(BO_target_problems, method_selection, max_seed):
    import pandas as pd
    n = len(BO_target_problems)

    median_data = []

    folders = ['bi_output']  #, 'bi_local_output']
    mean_cross_strategies = []
    for folder in folders:
        pname_list = []
        mean_smds = []
        for j in np.arange(0, n, 2):
            target_problem = BO_target_problems[j]
            target_problem = eval(target_problem)
            problem_name = target_problem.name()
            problem_name = problem_name[0:-2]
            pname_list.append(problem_name)
            # print(problem_name)
            working_folder = os.getcwd()
            result_folder = working_folder + '\\' + folder + '\\' + problem_name + '_' + method_selection

            n_fev = []
            for seed_index in range(max_seed):
                saveName = result_folder + '\\ll_nfev' + str(seed_index) + '.csv'
                data = np.loadtxt(saveName, delimiter=',')
                n_fev = np.append(n_fev, data)
            mean_nfev = np.median(n_fev)
            mean_smds = np.append(mean_smds, mean_nfev)

        mean_cross_strategies = np.append(mean_cross_strategies, mean_smds)


    mean_cross_strategies = np.atleast_2d(mean_cross_strategies).reshape(-1, 1)

    h = pd.DataFrame(mean_cross_strategies, columns=['lower level nfev'], index=pname_list)
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_process'
    saveName = result_folder + '\\nfev_median.csv'
    h.to_csv(saveName)


def results_process_before_after(BO_target_problems, method_selection, alg_folder, accuracy_name, seedmax):
    import pandas as pd
    n = len(BO_target_problems)
    mean_data = []
    median_data = []
    pname_list =[]
    for j in np.arange(0, n, 2):
        target_problem = BO_target_problems[j]
        target_problem = eval(target_problem)
        problem_name = target_problem.name()
        problem_name = problem_name[0:-2]
        pname_list.append(problem_name)
        # print(problem_name)
        working_folder = os.getcwd()
        result_folder = working_folder + '\\' + alg_folder + '\\' + problem_name + '_' + method_selection

        accuracy_data = []
        for seed_index in range(seedmax):
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

# component function of outer_process
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


# entrance for process before/after
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
    problem = problem_u.name()[0:-2]
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_' + method_selection

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    saveName = result_folder + '\\accuracy_before_reevaluation_' + str(seed_index) + '.csv'
    np.savetxt(saveName, s, delimiter=',')


def save_function_evaluation(nfev, problem, seed_index, method_selection, folder):
    working_folder = os.getcwd()
    problem = problem.name()[0:-2]
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

def compare_python_matlab():
    import pandas as pd

    pname = 'save_local_results.csv'
    mname = 'search_result.csv'
    pname_nfev = 'save_nfev.csv'
    mname_nfev = 'mfnev.csv'
    p = np.loadtxt(pname)
    m = np.loadtxt(mname)
    pfnev = np.loadtxt(pname_nfev)
    mfnev = np.loadtxt(mname_nfev)

    p = np.atleast_2d(p).reshape(-1, 1)
    p = np.vstack((p, np.median(p)))
    m = np.atleast_2d(m).reshape(-1, 1)
    m = np.vstack((m, np.median(m)))
    pfnev = np.atleast_2d(pfnev).reshape(-1, 1)
    pfnev = np.vstack((pfnev, np.median(pfnev)))
    mfnev = np.atleast_2d(mfnev).reshape(-1, 1)
    mfnev = np.vstack((mfnev, np.median(mfnev)))

    out = np.hstack((p, m, pfnev, mfnev))


    index = []
    for i in range(29):
        index = np.append(index, str(i))
    index = np.append(index, 'median')

    h = pd.DataFrame(out, columns=['p localsearch', 'm localsearch', 'p nfev', 'm nfev'], index=index)
    saveName = 'pm_compare.csv'
    h.to_csv(saveName)


def visualization_smd3(problem, seed_index):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    n_samples = 1000
    train_x, train_y, cons_y = init_xy(n_samples, problem, seed_index,
                                       **{'problem_type': 'bilevel'})
    x_other = [0., 0.]
    num_l = n_samples
    level = 'lower'
    x_other = np.atleast_2d(x_other)
    xother_expand = np.repeat(x_other, num_l, axis=0)
    if level == 'lower':
        complete_x = np.hstack((xother_expand, train_x))
    else:
        complete_x = np.hstack((train_x, xother_expand))
    train_y = problem.evaluate(complete_x, return_values_of=["F"])

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel('xl1_1')
    ax.set_xlim([-5, 10])
    ax.set_ylabel('xl1_2')
    ax.set_ylim([-5, 10])
    ax.set_zlabel('xl2_1');
    ax.set_zlim([-np.pi / 2, np.pi / 2])
    cm = plt.cm.get_cmap('RdYlBu')
    cl = ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c=train_y.ravel(), cmap=cm)
    ax.set_title('1000 points in xl boundaries')






    train_y[train_y < 3] = 0
    train_y_vio = train_y.sum(axis=1)
    throwaway = np.nonzero(train_y_vio)
    a = np.arange(0, n_samples)
    keep_index = np.setdiff1d(a, throwaway)

    train_y = train_y[keep_index, :]
    train_x = train_x[keep_index, :]
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlabel('xl1_1')
    ax.set_xlim([-5, 10])
    ax.set_ylabel('xl1_2')
    ax.set_ylim([-5, 10])
    ax.set_zlabel('xl2_1');
    ax.set_zlim([-np.pi / 2, np.pi / 2])
    cm = plt.cm.get_cmap('RdYlBu')
    ax.scatter(train_x[:, 0], train_x[:, 1], train_x[:, 2], c=train_y.ravel(), cmap=cm)
    fig.subplots_adjust(right=0.8)
    fig.colorbar(cl)


    ax.set_title('fl values smaller than 3')
    #ax.view_init(60, 10)
    plt.show()
    plt.savefig('smd3_l.eps', format='eps')

def plotCountour():
    # smd3
    xl1 = np.linspace(-5, 10, 100)
    xl2 = np.linspace(-np.pi/2 + 1e-8, np.pi/2 - 1e-8, 100)

    mesh_xl1, mesh_xl2 = np.meshgrid(xl1, xl2)
    fl = 1 + mesh_xl1**2 - np.cos(2 * np.pi * mesh_xl1) + np.tan(mesh_xl2) ** 2
    fl[fl > 10] = 11
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cs = ax.contourf(mesh_xl1, mesh_xl2, fl, levels=50, cmap="RdBu_r")
    fig.colorbar(cs, ax=ax)
    plt.show()

def test_if(a,b):
    a = (True, False) if a > b else (False, True)
    return zip(a)



if __name__ == "__main__":


    # problems_json = 'p/bi_problems'
    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    problem_u = eval(target_problems[0])

    # EGO_rebuildplot(problem_u, 'bi_output')




    # in general post process
    # ------------ result process--------------
    problems = target_problems
    folder = hyp['alg_settings']['folder']
    results_process_bestf(problems, 'eim', 11, folder)
    # combine_fev(problems, 'eim', 11)
    # results_process_before_after(problems, 'eim', 'bi_output', 'accuracy', 29)
    # --------------result process ------------


    '''
    # check with prblem BLTP5
    x_u = np.atleast_2d([17.0/9.0])
    # x_u = np.atleast_2d([1.80783])
    x_l = np.atleast_2d([[0.86488, 0.00]])
    problems = target_problems[0:2]
    target_problem_u = eval(problems[0])
    target_problem_l = eval(problems[1])

    eim_l = EI.EIM(target_problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=target_problem_l.xu,
                   lower_bound=target_problem_l.xl)

    matching_x, matching_f, n_fev_local, feasible_flag = \
        search_for_matching_otherlevel_x(x_u, 30, 20, target_problem_l, 'lower', eim_l, 100, 100, 0, False, 'eim')

    localsearch_xl = matching_x

    print('lowerlevel search returns feasibility: %d' % feasible_flag)

    localsearch_xl, localsearch_fl, local_fev = \
        hybridsearch_on_trueEvaluation(x_l, 'lower', x_u, target_problem_l)

    new_complete_x = np.hstack((np.atleast_2d(x_u), np.atleast_2d(localsearch_xl)))
    new_fl = target_problem_l.evaluate(new_complete_x, return_values_of=["F"])
    new_fu = target_problem_u.evaluate(new_complete_x, return_values_of=["F"])
    print('fu after re-valuation')
    print(new_fu)

    old_complete_x = np.hstack((np.atleast_2d(x_u), np.atleast_2d(x_l)))
    old_fu = target_problem_u.evaluate(old_complete_x, return_values_of=["F"])
    print('fu before re-valuation')
    print(old_fu)

    x_uopt = np.atleast_2d([17.0/9.0])
    x_lopt = np.atleast_2d([8/9, 0])
    x_best = np.hstack((x_uopt, x_lopt))
    fl_opt, gl_opt = target_problem_l.evaluate(x_best, return_values_of=["F", "G"])
    fu_opt = target_problem_u.evaluate(x_best, return_values_of=["F"])
    print('best upper level f: %.5f' % fu_opt)
    print('best lower level f: %.5f' % fl_opt)
    print('best lower level constrait')
    print(gl_opt) 
    '''


    '''
    from surrogate_problems import BLTP
    seed = 1
    np.random.seed(seed)
    target_problem_u = BLTP.BLTP9_F()  # p, r, q
    target_problem_l = BLTP.BLTP9_f()  # p, r, q

    xu, _, _ = init_xy(30, target_problem_u, seed, **{'problem_type': 'bilevel'})
    xl, _, _ = init_xy(30, target_problem_l, seed, **{'problem_type': 'bilevel'})
    # visualization_smd3(problem, 0)
    np.savetxt('testxu.csv', xu, delimiter=',')
    np.savetxt('testxl.csv', xl, delimiter=',')

    x = np.hstack((xu, xl))
    f, g = target_problem_l.evaluate(x, return_values_of=['F', 'G'])
    print(f)
    print(g)
    f, g = target_problem_u.evaluate(x, return_values_of=['F', 'G'])
    print(f)
    print(g)
    '''




