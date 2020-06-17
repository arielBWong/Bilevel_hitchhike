import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import optimizer_EI
from sklearn.utils.validation import check_array
import scipy
from scipy.optimize import SR1, BFGS
import pyDOE
from cross_val_hyperp import cross_val_krg
from surrogate_problems import branin, GPc, Gomez3, Mystery, Reverse_Mystery, SHCBc, HS100, Haupt_schewefel, \
                               MO_linearTest, single_krg_optim, WFG, iDTLZ, DTLZs, SMD, EI, Surrogate_test,BLTP

import os
import json
import copy
import joblib
import time
import multiprocessing as mp



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
    if len(kwargs) > 0:  # bilevel
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


# infeasible values are set 1.1 max f dynamically
# only for single objective
def feasibility_adjustment_3_dynamic(combine_y, feasibility):
    feasibility = np.array(feasibility)

    infeasible_index = np.argwhere(feasibility == False)
    feasible_index = np.argwhere(feasibility == True)

    infeasible_index = infeasible_index.ravel()
    feasible_index = feasible_index.ravel()

    if len(infeasible_index) > 0:
        max_y = np.max(combine_y[feasible_index, :]) * 1.2
        print('infeasible xl cause fu value adjustment: %4.4f' % max_y)
        # use current max feasible f value as infeasible punishment value
        combine_y[infeasible_index, :] = max_y
    else:
        print('all xl feasible, no need to modify')

    return combine_y


def search_for_matching_otherlevel_x(x_other, search_iter, n_samples, problem, level, eim, eim_pop, eim_gen,  seed_index, enable_crossvalidation, method_selection, **kwargs):

    np.set_printoptions(precision=16)
    # for testing SMD problem;
    D = problem.n_levelvar
    n_samples = 11 * D - 1

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
            # print('lower ego success to find feasible xl/fl for local search starting point is:')
            complete_y = complete_y_feasible
            train_x = train_x_feasible
            best_y_index = np.argmin(complete_y)
            best_x = train_x[best_y_index, :]
            best_x = np.atleast_2d(best_x)
            best_y = np.min(complete_y)
            # print(best_x)
            # print(best_y)
        else:
            # print("lower ego fail to find feasible, local search start with x: ")
            best_x, best_y = nofeasible_select(complete_c, complete_y, train_x)
            # print(best_x)

            # print("before local search, closest best y: %.4f" % best_y)
    else:
        best_y_index = np.argmin(complete_y)
        best_x = train_x[best_y_index, :]
        best_x = np.atleast_2d(best_x)
        best_y = np.min(complete_y)

    # conduct local search with true evaluation
    # print('from matching search...returned local search')
    # print('adding preprocessing, if starting point is too close to  bound move inwards...')
    best_x = boundary_closeness_check(best_x, problem.xu, problem.xl)


    localsearch_x, localsearch_f, success, n_fev, _, _, _, _ = \
        localsearch_on_trueEvaluation(best_x, 100, level, x_other, problem, None, None, None, None)
    n_fev = n_fev + search_iter + n_samples
    # print('local search success is : ')
    # print(success)



    if len(complete_y_feasible) > 0 and not success:
        best_x = boundary_closeness_check(best_x, problem.xu, problem.xl)
        localsearch_x, localsearch_f, success, n_fev, _, _, _, _ = \
            localsearch_on_trueEvaluation(best_x, 100, level, x_other, problem, None, None, None, None)
        n_fev = n_fev + search_iter + n_samples
        print('after adjusted xl start, local search status')
        print(success)
    if not success:
        print('not success, startx, endx, endf')
        print(best_x)
        print(localsearch_x)
        print(localsearch_f)



    # print('fl is ')
    # print(localsearch_f)
    # print('\n')

    # print('local search %s level, before %.4f, after %.4f' % (level, best_y, localsearch_f))
    # print('local search found lx')
    # print(localsearch_x)

    '''
    if problem.n_constr > 0:
        if len(complete_y_feasible) > 0:
            print('feasible exist after surrogate for upper xu ')
            print(x_other)
        else:
            print('no feasible after surrogate for upper xu %0.4f')
            print(x_other)
    '''


    # decide which x is returned
    if problem.n_constr > 0:
        # check feasibility of local search
        if level == 'lower':
            c = problem.evaluate(np.hstack((x_other, np.atleast_2d(localsearch_x))), return_values_of=["G"])
        else:
            c = problem.evaluate(np.hstack((np.atleast_2d(localsearch_x), x_other)), return_values_of=["G"])
        # print('local search return constraint:')
        # print(c)
        c[np.abs(c) < 1e-8] = 0

        # surrogate found feasible, consider local returns feasible
        # decide on smaller f value
        if len(complete_y_feasible) > 0:
            if np.any(c > 0):   # surrogate has feasible, local search has no
                # print('surrogate returns feasible, but local search not, return surrogate results')
                # print('surrogate constraint output is:')
                sc = problem.evaluate(np.hstack((x_other, np.atleast_2d(best_x))), return_values_of=["G"])
                # print(sc)
                return_tuple = (np.atleast_2d(best_x), np.atleast_2d(best_y), n_fev, True)
            else:
                # print('both surrogate and local search returns feasible, return smaller one')
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
                # print('local search also has no feasible solution, return false flag')
                feasible_flag = False
                return_tuple = (localsearch_x, localsearch_f, n_fev, feasible_flag)
            else:  # local search found feasible while surrogate did not
                # print('local search found feasible while surrogate did not')
                return_tuple = (localsearch_x, localsearch_f, n_fev, True)
    # no-constraint problem process
    else:
        return_tuple = (localsearch_x, localsearch_f, n_fev, True) \
            if localsearch_f < best_y \
            else (np.atleast_2d(best_x), np.atleast_2d(best_y), n_fev, True)

    return return_tuple[0], return_tuple[1], return_tuple[2], return_tuple[3]


def boundary_closeness_check(x, bound_u, bound_l):
    # this function checks with whether x is too close to bounds
    # if any of them is too close to boundaries, then change them a bit
    # further away from boundary
    # criterion is what ?
    x = check_array(x)
    upper_dist = np.abs(bound_u - x).ravel()
    up_ind = np.argwhere(upper_dist < 0.1).ravel()
    x[0, up_ind] = x[0, up_ind] - 0.1
    lower_dist = np.abs(x - bound_l).ravel()
    lower_ind = np.argwhere(lower_dist < 0.1).ravel()
    x[0, lower_ind] = x[0, lower_ind] + 0.1

    return x



def localsearch_for_matching_otherlevel_x(x_other, max_eval, search_level, problem, seed_index):
    # random init for x
    dimensions = problem.n_levelvar
    start_x = np.random.rand(1, dimensions)
    start_x = problem.xl + start_x * np.fabs(problem.xu - problem.xl)
    x_other = np.atleast_2d(x_other)
    start_x = np.atleast_2d(start_x)
    localsearch_x, f,_, nfev, _, _, _, _ = localsearch_on_trueEvaluation(start_x, max_eval, search_level, x_other, problem)
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

def hybridsearch_on_trueEvaluation(ankor_x, level, other_x, true_problem, ea_pop, ea_gen):

    if ankor_x is None:
        para = {
            "callback": bi_level_compensate_callback,
            "level": level,
            "other_x": other_x
        }
    else:
        para = {
            "add_info": ankor_x,
            "callback": bi_level_compensate_callback,
            "level": level,
            "other_x": other_x
        }

    bounds = np.vstack((true_problem.xl, true_problem.xu)).T.tolist()
    # print('before EA search')

    if ankor_x is not None:
        f, g = true_problem.evaluate(np.atleast_2d(np.hstack((other_x, ankor_x))), return_values_of=['F', 'G'])
        print('before revaluation fl value is: %.4f' % f)
        print('cons value is ')
        print(g)
        print('\n')

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
                               ea_pop,
                               ea_gen,
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
    if np.any(c > 0):
        print('EA returns infeasible ')
    else:
        print('EA returns feasible solutions')

    best_x, best_f, _, nfev, _, _, _, _ = \
        localsearch_on_trueEvaluation(best_x, 250, "lower", other_x, true_problem,
                                      None, None, None, None)
    nfev = nfev + ea_pop * ea_gen

    best_x = np.atleast_2d(best_x)
    other_x = np.atleast_2d(other_x)

    f, c = true_problem.evaluate(np.hstack((other_x, best_x)),  return_values_of=['F', 'G'])
    print('after local search true evaluation: ')
    print('fl value is: %.4f' % f)
    print('cons value is ')
    print(c)
    final_x = np.hstack((other_x, best_x))
    print('final_x')
    print(final_x)

    flag = True
    if true_problem.n_constr > 0:
        c[np.abs(c) <= 1e-6] = 0
        if np.any(c > 0):
            flag = False

    return best_x, best_f, nfev, flag

def ankor_selection(data_x, data_y, data_c, target_problem):
    # this method is for selection minimum f and its x
    # for both unconstraint and constraint problems

    if target_problem.n_constr > 0:
        y_feas, x_feas = return_feasible(data_c, data_y, data_x)
        y_feas = np.atleast_2d(y_feas).reshape(-1, target_problem.n_obj)
        x_feas = np.atleast_2d(x_feas).reshape(-1, target_problem.n_var)
        if len(y_feas) > 0:  # deal with feasible solutions
            # identify best feasible solution
            z = len(y_feas)
            print('after ankor selection, %d points left' % z)
            min_fu_index = np.argmin(y_feas)
            best_y = np.min(y_feas)
            # extract xu
            best_x_sofar = np.atleast_2d(x_feas[min_fu_index, 0:target_problem.n_levelvar])
        else:  # process situation where no feasible solutions
            print('no feasible solution found on given data')
            # nofeasible_select(constr_c, train_y, train_x):
            best_x, best_y = nofeasible_select(data_c, data_y, data_x)
            best_x_sofar = best_x[0, 0:target_problem.n_levelvar]
    else:  # process for uncontraint problems
        best_solution_index = np.argmin(data_y)
        best_y = np.min(data_y)
        best_x_sofar = data_x[best_solution_index, 0:target_problem.n_levelvar]

    best_x_sofar = np.atleast_2d(best_x_sofar)
    best_y = np.atleast_2d(best_y)   # totally redundant, but useful in ploting
    return best_x_sofar, best_y


def bilevel_localsearch(target_problem_u, complete_x_u, complete_y_u, complete_c_u, feasible_check, **bi_para):
    # this method is designed to include steps of running a local search on upper level varible
    # it selects the current best xu, and run a local search from this xu
    best_xu, best_fu = ankor_selection(complete_x_u, complete_y_u, complete_c_u, target_problem_u)
    best_xu_sofar = np.atleast_2d(best_xu)
    print('starting point')
    print(best_xu)
    print('starting fu')
    print(best_fu)

    new_xu, new_fu, success, n_fev_local, feasible_check, complete_x_u, complete_y_u, complete_c_u = \
        localsearch_on_trueEvaluation(best_xu_sofar,
                                      20,
                                      'upper',
                                      None,
                                      target_problem_u,
                                      complete_x_u,
                                      complete_y_u,
                                      complete_c_u,
                                      feasible_check,
                                      **bi_para)
    print('end point is :')
    print(new_xu)
    print('end point fu is: ')
    print(new_fu)
    return new_xu, n_fev_local, feasible_check, complete_x_u, complete_y_u, complete_c_u

def localsearch_on_trueEvaluation(ankor_x, max_eval, level_s, other_x, true_problem, method,
                                  complete_x_u,     # include existing samples for bilevel local search
                                  complete_y_u,     # include existing samples for bilevel local search
                                  complete_c_u,     # include existing samples for bilevel local search
                                  feasible_check,   # include feasible_check, this variable flag lower level infeasibility
                                  **bi_para):

    ankor_x = np.atleast_2d(ankor_x)
    if other_x is not None:
        other_x = np.atleast_2d(other_x)
    bi_matching_x = None
    up_evalcount = 0

    def obj_func(x):
        nonlocal feasible_check
        nonlocal up_evalcount
        nonlocal complete_x_u
        nonlocal complete_y_u
        nonlocal complete_c_u
        nonlocal true_problem
        nonlocal bi_matching_x

        x = np.atleast_2d(x)
        if other_x is not None:  # means single level optimization
            if level_s == 'lower':
                x = np.hstack((other_x, x))
            else:
                x = np.hstack((x, other_x))
            f = true_problem.evaluate(x,  return_values_of=["F"])

            '''
            if other_x is not None:
                print('\n Single level  calculate obj')
                print('Single fl')
                print(f)
                print('single x')
                print(x)
            '''


            return f.ravel()
        else:  # indicate a local search on upper level/bilevel optimization wrapper
            # the local search optimization framework seems calculate constraint first, so
            # should wait for contraint update the xl
            # but also have to be compatible if upper level is unconstraint problem
            if true_problem.n_constr > 0:
                # assume contraint objective function has calculated
                # bilevel_match xl
                bi_matching_x = np.atleast_2d(bi_matching_x)  # double security
                x = np.hstack((x, bi_matching_x))
                f, c = true_problem.evaluate(x, return_values_of=["F", "G"])
            else:  # process for unconstraint problem
                matching_xl, matching_fl, n_fev_local, feasible_flag = \
                    search_for_matching_otherlevel_x(x, **bi_para)
                up_evalcount = up_evalcount + n_fev_local
                # bilevel local search only happens on upper level
                # so combination of variables has only one  form
                bi_matching_x = np.atleast_2d(matching_xl)  # double security
                x = np.hstack((x, bi_matching_x))
                f, c = true_problem.evaluate(x, return_values_of=["F", "G"])

                # if lower level return infeasible solution xl
                # f value of upper level needs to be changed to penalty values
                # double check with feasibility returned from other level
                feasible_check = np.append(feasible_check, feasible_flag)

                # adding new xu yu to training
                complete_x_u = np.vstack((complete_x_u, x))
                complete_y_u = np.vstack((complete_y_u, f))
                complete_c_u = np.vstack((complete_c_u, c))
                if feasible_flag is False:
                    complete_y_u = feasibility_adjustment_3_dynamic(complete_y_u, feasible_check)
                    f = np.atleast_2d(complete_y_u[-1, :])

                #  n = complete_x_u.shape[0]
                # print('local search changing check: %d'% n)
                # print('return f: %d' % f)

            '''
            if other_x is None:

                print('\n bilevel calculate obj')
                print('bilevel each fu')
                print(f)
                print('bilevel xu')
                print(x)
                print('\n')
            '''

            return f.ravel()

    def obj_func_tr(x):  # for trust region
        nonlocal feasible_check
        nonlocal up_evalcount
        nonlocal complete_x_u
        nonlocal complete_y_u
        nonlocal complete_c_u
        nonlocal true_problem
        nonlocal bi_matching_x

        x = np.atleast_2d(x)
        if other_x is not None:  # means single level optimization
            if level_s == 'lower':
                x = np.hstack((other_x, x))
            else:
                x = np.hstack((x, other_x))
            f = true_problem.evaluate(x,  return_values_of=["F"])

            '''
            if other_x is not None:
                print('\n Single level  calculate obj')
                print('Single fl')
                print(f)
                print('single x')
                print(x)
            '''
            return f.ravel()
        else:  # indicate a local search on upper level/bilevel optimization wrapper

            print('bilevel obj calculation with lower search:')
            print(x)
            matching_xl, matching_fl, n_fev_local, feasible_flag = \
                search_for_matching_otherlevel_x(x, **bi_para)
            up_evalcount = up_evalcount + n_fev_local
            # bi-level local search only happens on upper level
            # so combination of variables has only one form
            bi_matching_x = np.atleast_2d(matching_xl)  # double security


            x = np.hstack((x, bi_matching_x))
            f, c = true_problem.evaluate(x, return_values_of=["F", "G"])
            print(f)

            # if lower level return infeasible solution xl
            # f value of upper level needs to be changed to penalty values
            # double check with feasibility returned from other level
            feasible_check = np.append(feasible_check, feasible_flag)

            # adding new xu yu to training
            complete_x_u = np.vstack((complete_x_u, x))
            complete_y_u = np.vstack((complete_y_u, f))
            complete_c_u = np.vstack((complete_c_u, c))
            if feasible_flag is False:
                complete_y_u = feasibility_adjustment_3_dynamic(complete_y_u, feasible_check)
                f = np.atleast_2d(complete_y_u[-1, :])

                #  n = complete_x_u.shape[0]
                # print('local search changing check: %d'% n)
                # print('return f: %d' % f)

            '''
            if other_x is None:

                print('\n bilevel calculate obj')
                print('bilevel each fu')
                print(f)
                print('bilevel xu')
                print(x)
                print('\n')
            '''

            return f.ravel()

    def cons_func(x):

        nonlocal feasible_check
        nonlocal up_evalcount
        nonlocal complete_x_u
        nonlocal complete_y_u
        nonlocal complete_c_u
        nonlocal true_problem
        nonlocal bi_matching_x

        x = np.atleast_2d(x)
        if other_x is not None:  # means single level optimization
            if level_s == 'lower':
                x = np.hstack((other_x, x))
            else:
                x = np.hstack((x, other_x))

            constr = true_problem.evaluate(x, return_values_of=["G"])
        else:  # deal with bi-level local search condition
            print('bilevel wrapper lower matching search for xu:')
            print(x)
            matching_xl, matching_fl, n_fev_local, feasible_flag = \
                search_for_matching_otherlevel_x(x, **bi_para)
            up_evalcount = up_evalcount + n_fev_local
            # bi-level local search only happens on upper level
            # so combination of variables has only one form
            bi_matching_x = np.atleast_2d(matching_xl)  # double security
            x = np.hstack((x, bi_matching_x))

            F, constr = true_problem.evaluate(x, return_values_of=["F", "G"])


            # if lower level return infeasible solution xl
            # f value of upper level needs to be changed to penalty values
            # double check with feasibility returned from other level
            feasible_check = np.append(feasible_check, feasible_flag)

            # adding new xu yu to training samples
            complete_x_u = np.vstack((complete_x_u, x))
            complete_y_u = np.vstack((complete_y_u, F))
            complete_c_u = np.vstack((complete_c_u, constr))
            if feasible_flag is False:
                complete_y_u = feasibility_adjustment_3_dynamic(complete_y_u, feasible_check)

        '''
        if other_x is None:
            print('bilevel local search on contraints')
            print(constr)
            print('from x combo:')
            print(x)
            print('\n')
        if other_x is not None:
            print('\n single level local search on contraints:')
            print(constr)
            print('from x combo:')
            print(x)
        '''

        constr = constr * -1
        return constr.ravel()

    def cons_func_tr(x):  # for trust region

        nonlocal feasible_check
        nonlocal up_evalcount
        nonlocal complete_x_u
        nonlocal complete_y_u
        nonlocal complete_c_u
        nonlocal true_problem
        nonlocal bi_matching_x

        x = np.atleast_2d(x)
        if other_x is not None:  # means single level optimization
            if level_s == 'lower':
                x = np.hstack((other_x, x))
            else:
                x = np.hstack((x, other_x))

            constr = true_problem.evaluate(x, return_values_of=["G"])
        else:  # deal with bi-level local search condition
            # assume obj has calculated bi_matching_x
            print('bilevel calculate constraint')
            x = np.hstack((x, bi_matching_x))
            # print(x)
            F, constr = true_problem.evaluate(x, return_values_of=["F", "G"])

        '''
        if other_x is None:
            print('bilevel local search on contraints')
            print(constr)
            print('from x combo:')
            print(x)
            print('\n')
        if other_x is not None:
            print('\n single level local search on contraints:')
            print(constr)
            print('from x combo:')
            print(x)
        '''

        # constr = constr * -1
        return constr.ravel()

    bounds = scipy.optimize.Bounds(lb=true_problem.xl, ub=true_problem.xu)


    if method == 'trust-constr':
        ankor_x = ankor_x.ravel()
        if true_problem.n_constr > 0:
            nc = scipy.optimize.NonlinearConstraint(cons_func_tr, [-np.inf, -np.inf], [0, 0], jac='2-point', hess=SR1())
            optimization_res = scipy.optimize.minimize(
                obj_func_tr, ankor_x, method="trust-constr", options={'maxiter': max_eval},
                constraints=nc, bounds=bounds)
        else:
            optimization_res = scipy.optimize.minimize(
                obj_func_tr, ankor_x, method="trust-constr", options={'maxiter': max_eval}, jac=False,
                bounds=bounds)
    if method == 'slsqp':
        if true_problem.n_constr > 0:
            optimization_res = scipy.optimize.minimize(obj_func, ankor_x, method='slsqp',
                                                       options={'maxiter': max_eval, 'disp': 3, 'iprint': 3},
                                                       constraints={'type': 'ineq', 'fun': cons_func},
                                                       bounds=bounds,
                                                       tol=1e-10)
        else:
            optimization_res = scipy.optimize.minimize(obj_func, ankor_x, method='slsqp',
                                                       options={'maxiter': max_eval},
                                                       bounds=bounds)


    # print('number of function evaluations: %d '% optimization_res.nfev)

    x, f, num_fev, success = optimization_res.x, optimization_res.fun, optimization_res.nfev, optimization_res.success


    if other_x is None:
        print('\n bilevel returned results')
        print(optimization_res)

    '''
    else:
        print('single level local search result')
        print(optimization_res)
    '''


    return x, f, success, num_fev + up_evalcount,  feasible_check, complete_x_u, complete_y_u, complete_c_u





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


def dynamic_close_point_elimination(train_x_org, train_y_org, train_c_org, measure):
    # measure is a list int numbers [1, 2, 3]
    max_digi = measure[-1]
    n_obj = train_y_org.shape[1]
    n_var = train_x_org.shape[1]
    if train_c_org is not None:
        n_cons = train_c_org.shape[1]
    else:
        n_cons = 0

    mse_collection = []
    krg_collection = []
    xmean_collection = []
    xstd_collection = []

    mse_g_collection = []
    krg_g_collection = []
    for l in measure:
        # data with _org do not change
        train_x, train_y, train_c = close_point_elimination(train_x_org, train_y_org, train_c_org, l)
        # train_x = train_x_orig
        # train_y = train_y_orig
        # train_c = train_c_orig
        train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
        train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

        xmean_collection = np.append(xmean_collection, x_mean)
        xstd_collection = np.append(xstd_collection, x_std)

        train_x_norm = np.atleast_2d(train_x_norm)
        train_y_norm = np.atleast_2d(train_y_norm)

        if train_c is not None:
            train_c_norm, c_mean, c_std = norm_by_zscore(train_c)
            train_c_norm = np.atleast_2d(train_c_norm)
        else:
            train_c_norm = None

        (krg, krg_mse), (krg_g, krg_g_mse) = cross_val_krg(train_x_norm, train_y_norm, train_c_norm, True)
        mse_collection = np.append(mse_collection, krg_mse)  # becomes np.array
        krg_collection = np.append(krg_collection, krg)      # becomes np.array, so needs reshape

        mse_g_collection = np.append(mse_g_collection, krg_g_mse)
        krg_g_collection = np.append(krg_g_collection, krg_g)

    mse_collection = np.atleast_2d(mse_collection).reshape(max_digi, n_obj)
    # get min mse from each objectives/constraints
    # mse_index = np.argmin(mse_collection, axis=0)

    xmean_collection = np.atleast_2d(xmean_collection).reshape(-1, n_var)
    xstd_collection = np.atleast_2d(xstd_collection).reshape(-1, n_var)

    krg_collection = np.atleast_2d(krg_collection).reshape(-1, n_obj)
    # this following 2 lines select different distance accuracy for different krging
    # but does not fit for EI process
    # a = [int(b) for b in np.linspace(0, n_obj-1, n_obj)]
    # selected_krg = krg_collection[mse_index, a]

    if train_c_org is not None:
        mse_g_collection = np.atleast_2d(mse_g_collection).reshape(max_digi, n_cons)
        # get min mse from each objectives/constraints
        # mse_g_index = np.argmin(mse_g_collection, axis=0)
        krg_g_collection = np.atleast_2d(krg_g_collection).reshape(-1, n_cons)

        # this following 2 lines select different distance accuracy for different krging
        # but does not fit for EI process
        # a = [int(b) for b in np.linspace(0, n_cons - 1, n_cons)]
        # selected_krg_g = krg_g_collection[mse_g_index, a]
    # else:
        # selected_krg_g = None

    # the reason to use average mse
    # is that following this krg training is EI process
    # It requires x mean and x std to bound sampling range
    # Now I am not changing the EI process so krg and krg_g
    # need to use the same normalization
    if train_c_org is not None:
        combo_mse = np.hstack((mse_collection, mse_g_collection))
    else:
        combo_mse = mse_collection
    # each column(kriging mse) is not at the same scale
    # need normalization
    combo_mse = normalization_with_self(combo_mse)
    combo_mse = np.sum(combo_mse, axis=1)
    dist_selection_index = np.argmin(combo_mse)
    selected_krg = krg_collection[dist_selection_index, :]
    if train_c_org is not None:
        selected_krg_g = krg_g_collection[dist_selection_index, :]
    else:
        selected_krg_g = []

    selected_x_mean = xmean_collection[dist_selection_index, :]
    selected_x_std = xstd_collection[dist_selection_index, :]
    # print('selected x mean')
    # print(selected_x_mean)

    return selected_krg, selected_krg_g, dist_selection_index








def surrogate_search_for_nextx(train_x_orig, train_y_orig, train_c_orig, eim, eim_pop, eim_gen, method_selection, enable_crossvalidation):

    #------ old version on fixed elimination-----------
    train_x, train_y, train_c = close_point_elimination(train_x_orig, train_y_orig, train_c_orig, 2)
    # train_x = train_x_orig
    # train_y = train_y_orig
    # train_c = train_c_orig
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
    # -----------old version on fixed elimination------------


    '''
    #---- dynamic process determine which distance to use ---------
    krg, krg_g, d_index = \
       dynamic_close_point_elimination(train_x_orig, train_y_orig, train_c_orig, [1, 2, 3])
    train_x, train_y, train_c = close_point_elimination(train_x_orig, train_y_orig, train_c_orig, d_index+1)

    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)
    # print('recreated x mean ')
    # print(x_mean)
    # print('\n')

    train_x_norm = np.atleast_2d(train_x_norm)
    train_y_norm = np.atleast_2d(train_y_norm)

    if train_c is not None:
        train_c_norm, c_mean, c_std = norm_by_zscore(train_c)
        train_c_norm = np.atleast_2d(train_c_norm)
    else:
        train_c_norm = None
    #---------  dynamic process determine which distance to use ---------
    '''


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

def trained_model_prediction(krg, train_x, train_y, test_x, n_var):
    # this function is a reusable one
    # for using a trained kriging to
    # predict test data
    # train_x, train_y, _ = close_point_elimination(train_x_org, train_y_org, None)
    # n = train_x.shape[0]
    # print('after rebuild number of data: %d' % n)
    # just for test
    # train_x = train_x_org
    # train_y = train_y_org

    # find mean and std for prediction
    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

    test_x = np.atleast_2d(test_x).reshape(-1, n_var)
    test_x_norm = norm_by_exist_zscore(test_x, x_mean, x_std)
    pred_y_norm, pred_y_sig_norm = krg.predict(test_x_norm)
    pred_y = reverse_with_zscore(pred_y_norm, y_mean, y_std)

    # test train x train y
    pred_ty_norm, pred_ty_sig_norm = krg.predict(train_x_norm)
    pred_ty = reverse_with_zscore(pred_ty_norm, y_mean, y_std)
    print(pred_ty - train_y)


    return pred_y





def ego_basic_train_predict(krg, krg_g, train_x, train_y, train_c, test_x, test_y, infeasible, problem_u, folder):
    # this method is only a service function for the  method
    # rebuild_surrogate_and_plot()
    # only works for one krg


    # find mean and std for prediction
    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)


    test_x_norm = norm_by_exist_zscore(test_x, x_mean, x_std)
    pred_y_norm, pred_y_sig_norm = krg.predict(test_x_norm)
    pred_y = reverse_with_zscore(pred_y_norm, y_mean, y_std)


    if train_c is not None:
        train_c_norm, c_mean, c_std = norm_by_zscore(train_c)
        train_c_norm = np.atleast_2d(train_c_norm)

        n_g = len(krg_g)
        pred_c = []
        for j in range(n_g):
            cons_norm = krg_g[j].predict(test_x_norm)
            cons = reverse_with_zscore(cons_norm, c_mean, c_std)
            pred_c = np.append(pred_c, cons)

        pred_c = np.atleast_2d(pred_c).reshape(-1, n_g, order='F')

        # now use pred_c to adjust feasibility of pred_y
        infeas_pair = []
        n_x = train_x.shape[0]
        for j in range(n_x):
            cons_y = np.atleast_2d(pred_c[j, :]).reshape(1, -1)  # check feasibility one by one
            cons_y[np.abs(cons_y) < 1e-10] = 0
            if np.any(cons_y) > 0:   # mark infeasible
                pair = [test_x[j, 0], pred_y[j, 0]]   # x has to be one variable, y
                infeas_pair = np.append(infeas_pair, pair)
            else:
                continue
        infeas_pair = np.atleast_2d(infeas_pair).reshape(-1, 2)
    else:
        train_c_norm = None



    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('design variable')
    ax1.set_ylabel('f and predicted f value')
    # comment following line only for test/ should comment
    ax1.scatter(test_x.ravel(), pred_y.ravel(), marker='|', c='g')
    ax1.scatter(test_x.ravel(), test_y, c='r')
    # ax1.fill_between(test_x.ravel(), (pred_y + pred_y_sig_norm).ravel(), (pred_y - pred_y_sig_norm).ravel(), alpha=0.5)
    ax1.scatter(train_x, train_y, marker='x')


    if train_c is not None:
        infeas_x = infeas_pair[:, 0]
        infeas_y = infeas_pair[:, 1]
        ax1.scatter(infeas_x, infeas_y, c='k')
        ax1.legend(['EGO kriging', 'exghaustive search', 'training', 'infeasible'])
        # for test-------
        # inf_testx = test_x[infeasible, :]
        # inf_testy = test_y[infeasible, :]
        # ax1.scatter(inf_testx, inf_testy, c='k')
        # for test-------

    else:
        ax1.legend(['EGO kriging', 'exghaustive search', 'training'])

    plt.title(problem_u.name()[0:-2])

    # save back to where krg model was saved
    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_krgmodels'
    plotsave = result_folder + '\\upper.png'
    plt.savefig(plotsave, format='png')

    # plt.show()


def get_infeasible_subset(data_x, data_y, data_c):
    # this function extracts infeasible subset from x-y pair
    data_x = check_array(data_x)
    data_y = check_array(data_y)
    data_c = check_array(data_c)

    data_g = copy.deepcopy(data_c)
    data_g[np.abs(data_g) < 1e-10] = 0
    data_g[data_g <= 0] = 0
    data_cg = data_g.sum(axis=1)
    infeasible = np.nonzero(data_cg)

    x_infeasible = np.atleast_2d(data_x[infeasible, :]).reshape(-1, 1)
    y_infeasible = np.atleast_2d(data_y[infeasible, :]).reshape(-1, 1)

    infeas_pair = np.hstack((x_infeasible, y_infeasible))

    return infeas_pair


def upper_y_from_exghaustive_search(problem_l, problem_u, xu_list, vio_value):
    #  this function is only a service function for
    #  rebuild_surrogate_and_plot()
    fu = []

    n = xu_list.shape[0]
    xl = []
    vio_list = []
    for i in range(n):
        xu = np.atleast_2d(xu_list[i, :])
        localsearch_xl, localsearch_fl, local_fev, flag = \
            hybridsearch_on_trueEvaluation(None, 'lower', xu, problem_l, 100, 100)

        combo_x = np.hstack((xu, localsearch_xl))
        fu_i = problem_u.evaluate(combo_x, return_values_of=["F"])

        xl = np.append(xl, localsearch_xl)

        if flag is False:
            if vio_value is not None:
                fu_i = vio_value
            else:
                fu_i = 1000
            vio_list = np.append(vio_list, False)
            print('False lower')
        else:
            vio_list = np.append(vio_list, True)
        fu = np.append(fu, fu_i)


    if vio_value is None:
        vio_list_False = np.argwhere(vio_list == 0).flatten()
        vio_list_True = np.argwhere(vio_list > 0).flatten()
        vio_list_True = [int(c) for c in vio_list_True]
        vio_list_False = [int(c) for c in vio_list_False]

        fu_True_max = np.max(fu[vio_list_True])
        fu[vio_list_False] = fu_True_max*1.2

    fu = np.atleast_2d(fu).reshape(-1, problem_u.n_obj)
    xl = np.atleast_2d(xl).reshape(-1, problem_l.n_levelvar)
    return fu, xl, vio_list

def plot_sample_order(train_x, train_y, problem_u, seed):
    # this function plots the value of x to y in the order of samples
    # train_x and train_y is assumed np.2d
    # this function only process train_x as one variable vector
    # same for train_y
    problem = problem_u.name()[0:-2]
    n = train_x.shape[0]
    color_order = np.linspace(0, n, n)


    fig = plt.figure()
    cm1 = plt.cm.get_cmap('RdYlBu')
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('design variable')
    ax1.set_ylabel('f and predicted f value')
    sc1 = ax1.scatter(train_x.ravel(), train_y.ravel(), c=color_order, cmap=cm1)
    fig.colorbar(sc1, ax=ax1)
    plt.title(problem + ' ' + str(n) + ' samples')

    working_folder = os.getcwd()
    result_folder = working_folder + '\\order_of_samples'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    plotsave = result_folder + '\\' + problem + '_sample_order_plot_' + str(seed) + '.png'
    plt.savefig(plotsave, format='png')

    # plt.show()



def rebuild_surrogate_and_plot():
    # investigation function
    # intend to find out how final surrogate perform in
    # rebuilding the real function after all the iteration
    # is finished
    # only work for upper level problem
    # only work for one upper variable problem

    # this function has dual outcome
    # comment the last line, if only want to
    # use this method to save test data

    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    folder ='bi_output'
    n = len(target_problems)
    # seedlist = [2, 0, 5, 8, 1, 6, 5, 6, 5, 6, 9]   # median seeds from no sample in infeasible regions
    # seedlist = [2, 0, 5, 1, 8, 6, 5, 6, 4, 1, 9]    # median seeds from dynamic values from infeasible regions
    seedlist = [4, 3, 4, 4]
    # seed_index = 10
    seed_index = 0

    for i in range(0, n, 2):
    # for i in range(0, 2, 2):
    # for i in range(8, 10, 2):  # only test problem 5
        problem_u = eval(target_problems[i])
        problem_l = eval(target_problems[i+1])
        # seed_index = 4  # only for test problem 5
        seed = seedlist[seed_index]
        print(problem_u.name())

        # this plot only works with single upper level  variable problems
        # if problem_u.n_levelvar > 1:
        #     seed_index = seed_index + 1
        #     continue

        # load the krg and train sample of each problem
        problem = problem_u.name()[0:-2]
        working_folder = os.getcwd()
        result_folder = working_folder + '\\' + folder + '\\' + problem + '_krgmodels'
        krgmodel_save = result_folder + '\\krg_' + str(seed) + '.joblib'
        print(krgmodel_save)
        krg = joblib.load(krgmodel_save)
        krgmodel_save = result_folder + '\\krg_g_' + str(seed) + '.joblib'
        print(krgmodel_save)
        krg_g = joblib.load(krgmodel_save)

        seed_index = seed_index + 1

        if len(krg) > 1:
            print('can only process single objective, skip')
            continue

        # use krg model to get predicted data
        # load training data
        result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
        traindata_file = result_folder + '\\sampled_data_x_' + str(seed)+'.csv'
        print(traindata_file)
        x_both = np.loadtxt(traindata_file, delimiter=',')
        traindata_file = result_folder + '\\sampled_data_y_' + str(seed)+'.csv'
        print(traindata_file)
        y_up = np.loadtxt(traindata_file, delimiter=',')

        # extract upper level variable
        train_x = np.atleast_2d(x_both[:, 0:problem_u.n_levelvar]).reshape(-1, problem_u.n_levelvar)
        train_y = np.atleast_2d(y_up).reshape(-1, 1)

        # plot_sample_order(train_x, train_y, problem_u, seed)

        # because of saving sequence problem, the last one needs to be deleted
        # as it cannot be counted into training, otherwise, the mean for
        # building prediction will be affected.
        m = train_x.shape[0]
        train_x = np.delete(train_x, m-1, axis=0)
        train_y = np.delete(train_y, m - 1, axis=0)

        # train_c is not saved but can be rebuilt
        train_c = problem_u.evaluate(np.atleast_2d(x_both), return_values_of=['G'])
        if train_c is not None:
            train_c = np.delete(train_c, m-1, axis=0)

        # identify the violation value is set to what
        feasiflag_save = result_folder + '\\sampled_ll_feasi_' + str(seed) + '.joblib'
        print(feasiflag_save)
        feasible_flag = joblib.load(feasiflag_save)
        feasible_flag = feasible_flag[0:-1]
        vio_list = np.argwhere(feasible_flag == False)
        if len(vio_list) > 0:
            vio_setting = train_y[vio_list[0], :]
        else:
            vio_setting = None

        # create test data from variable bounds
        testdata = np.linspace(problem_u.xl, problem_u.xu, 1000)
        testdata_y, xl, _ = upper_y_from_exghaustive_search(problem_l, problem_u, testdata, vio_setting)

        # save these test data
        result_folder = working_folder + '\\' + folder + '\\real_function'
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

        savefile = result_folder + '\\' + problem + 'testdate_gen_seed_' + str(seed) + '.csv'

        testdata = np.atleast_2d(testdata).reshape(-1, 1)
        testdata_y = np.atleast_2d(testdata_y).reshape(-1, 1)
        save_test = np.hstack((testdata, xl, testdata_y))  # save both level variables
        np.savetxt(savefile, save_test, delimiter=',')

        # test: violation on upper
        complete_x = np.hstack((testdata, xl))
        upper_c = problem_u.evaluate(complete_x, return_values_of=['G'])
        upper_c[np.abs(upper_c) < 1e-10] = 0
        all_g = copy.deepcopy(upper_c)

        all_g[all_g <= 0] = 0
        all_cv = all_g.sum(axis=1)
        infeasible = np.nonzero(all_cv)
        a = np.linspace(0, 1000, 1000, dtype=int)
        feasible = np.setdiff1d(a, infeasible)

        # comment the following function, if only want to save test data
        ego_basic_train_predict(krg[0], krg_g, train_x, train_y, train_c, testdata, testdata_y, infeasible, problem_u, folder)


def SMD_invest_retrieve_vio(problem_u, folder, seed):
    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()

    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    traindata_file = result_folder + '\\sampled_data_x_' + str(seed) + '.csv'
    print(traindata_file)
    x_both = np.loadtxt(traindata_file, delimiter=',')
    traindata_file = result_folder + '\\sampled_data_y_' + str(seed) + '.csv'
    print(traindata_file)
    y_up = np.loadtxt(traindata_file, delimiter=',')

    # extract upper level variable
    train_x = np.atleast_2d(x_both[:, 0:problem_u.n_levelvar]).reshape(-1, problem_u.n_levelvar)
    train_y = np.atleast_2d(y_up).reshape(-1, 1)


    # because of saving sequence problem, the last one needs to be deleted
    # as it cannot be counted into training, otherwise, the mean for
    # building prediction will be affected.
    m = train_x.shape[0]
    train_x = np.delete(train_x, m - 1, axis=0)
    train_y = np.delete(train_y, m - 1, axis=0)

    # identify the violation value is set to what
    feasiflag_save = result_folder + '\\sampled_ll_feasi_' + str(seed) + '.joblib'
    print(feasiflag_save)
    feasible_flag = joblib.load(feasiflag_save)
    feasible_flag = feasible_flag[0:-1]
    vio_list = np.argwhere(feasible_flag == False)
    if len(vio_list) > 0:
        vio_setting = train_y[vio_list[0], :]
    else:
        vio_setting = None
    return vio_setting


def test_data_parall(testdata,problem_u, problem_l, result_folder, seed, vio_setting):
    # this function split test data into 10 groups and run parallel
    # and only deal with 1000 instances

    args = []
    n = testdata.shape[0]
    n_var = testdata.shape[1]
    m = 10
    batch = 100
    for i in range(m):
        head = i * batch
        tail = (i+1) * batch
        eachdata = np.atleast_2d(testdata[head:tail, 0: n_var]).reshape(-1, n_var)
        args.append((i, eachdata, problem_u, problem_l, result_folder, seed, vio_setting))
    return args

def parallelsave_combine(problem_u, seed, batch_max):
    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'


    result_folder = working_folder + '\\' + folder + '\\real_function'
    out = [0] * (problem_u.n_var + 2)
    out = np.atleast_2d(out)
    for i in range(batch_max):
        savefile = result_folder + '\\' + problem + \
               '_testdate_sur_seed_' + str(seed) + '_' + str(i) + '.joblib'
        data = joblib.load(savefile)
        out = np.vstack((out, data))

    out = np.delete(out, 0, axis=0)
    return out


def surrogate_react_2_testdata(batch, data, problem_u, problem_l, result_folder, seed, infea):
    problem = problem_u.name()[0:-2]
    # create surrogat
    eim_l = EI.EIM(problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=problem_l.xu,
                   lower_bound=problem_l.xl)
    xl = []
    fl = []
    feasible_list = []
    n = data.shape[0]
    for i in range(n):
        print(i)
        xu = np.atleast_2d(data[i, :])
        matching_x, matching_f, n_fev_local, feasible_flag = \
            search_for_matching_otherlevel_x(xu,
                                             30,
                                             20,
                                             problem_l,
                                             'lower',
                                             eim_l,
                                             100,
                                             100,
                                             0,
                                             False,
                                             'eim')
        xl = np.append(xl, matching_x)
        fl = np.append(fl, matching_f)
        feasible_list = np.append(feasible_list, feasible_flag)

    xl = np.atleast_2d(xl).reshape(-1, problem_l.n_levelvar)
    fl = np.atleast_2d(fl).reshape(-1, problem_l.n_obj)

    data = np.atleast_2d(data)
    x_comb = np.hstack((data, xl))
    fu = problem_u.evaluate(x_comb, return_values_of=['F'])
    infesi_index = np.where(feasible_list == 0)[0]
    if infea is not None:
        fu[infesi_index, :] = infea
    else:
        fu[infesi_index, :] = 1000



    savefile = result_folder + '\\' + problem + \
               '_testdate_sur_seed_' + str(seed) + '_' + str(batch) + '.joblib'
    save_test = np.hstack((data, xl, fu, fl))
    joblib.dump(save_test, savefile)


def SMD_invest(problem_u, problem_l, vio_setting, seed):
    # this function aims to form the plot
    # compare lower f of exghausive search and surrogate search
    #

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'


    # save these test data
    result_folder = working_folder + '\\' + folder + '\\real_function'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    vio_setting = SMD_invest_retrieve_vio(problem_u, 'bi_output', 0)

    # create test data from variable bounds
    # testdata = np.linspace(problem_u.xl, problem_u.xu, 1000)
    testdata, _, _ = init_xy(1000, problem_u, 0, **{'problem_type': 'bilevel'})
    a = 0

    testdata_y, xl, vio_list = upper_y_from_exghaustive_search(problem_l, problem_u, testdata, vio_setting)

    savefile = result_folder + '\\' + problem + '_testdate_exh_seed_' + str(seed) + '.csv'

    testdata = np.atleast_2d(testdata).reshape(-1, problem_u.n_levelvar)
    testdata_y = np.atleast_2d(testdata_y).reshape(-1, problem_u.n_obj)
    save_test = np.hstack((testdata, xl, testdata_y))  # save both level variables
    np.savetxt(savefile, save_test, delimiter=',')

    '''


    # create surrogat
    eim_l = EI.EIM(problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=problem_l.xu,
                   lower_bound=problem_l.xl)
    xl = []
    fl = []

    # let us parallel it
    for i in range(10):
        print(i)
        xu = np.atleast_2d(testdata[i, :])
        matching_x, matching_f, n_fev_local, feasible_flag = \
            search_for_matching_otherlevel_x(xu,
                                             30,
                                             20,
                                             problem_l,
                                             'lower',
                                             eim_l,
                                             100,
                                             100,
                                             0,
                                             False,
                                             'eim')


        xl = np.append(xl, matching_x)
        fl = np.append(fl, matching_f)
    xl = np.atleast_2d(xl).reshape(-1, problem_l.n_levelvar)
    fl = np.atleast_2d(fl).reshape(-1, problem_l.n_obj)
    testdata = np.atleast_2d(testdata)
    x_comb = np.hstack((testdata, xl))
    testdata_y = problem_u.evaluate(x_comb, return_values_of=['F'])


    savefile = result_folder + '\\' + problem + '_testdate_sur_seed_' + str(seed) + '.csv'
    testdata = np.atleast_2d(testdata).reshape(-1, problem_u.n_levelvar)
    testdata_y = np.atleast_2d(testdata_y).reshape(-1, problem_u.n_obj)

    save_test = np.hstack((testdata, xl, testdata_y, fl))

    np.savetxt(savefile, save_test, delimiter=',')
     '''

def meshgrid_distribute_data():
    print('to be continued')

def invest_SMD_one_instance(problem_u, problem_l, index, seed):
    # this function check the lower level search for designated problem
    # and seed

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'


    # In order to plot the test data, we need the training data of krg
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    traindata_file = result_folder + '\\sampled_data_x_' + str(seed) + '.csv'
    print(traindata_file)
    x_both = np.loadtxt(traindata_file, delimiter=',')
    traindata_file = result_folder + '\\sampled_data_y_' + str(seed) + '.csv'
    print(traindata_file)
    y_up = np.loadtxt(traindata_file, delimiter=',')

    xu = np.atleast_2d(x_both[index, 0:problem_u.n_levelvar])
    x_combo = np.atleast_2d(x_both[index, 0:problem_u.n_var])
    xl = np.atleast_2d(x_both[index, problem_u.n_levelvar:problem_u.n_var])
    xl[0, 0] = xl[0, 0] + 0.01
    xl[0, 1] = xl[0, 1] + 0.01
    xl[0, 2] = xl[0, 2] - 0.01
    # for test purpose
    # xl = np.atleast_2d(x_both[index, [problem_u.n_levelvar, problem_u.n_var-1]])
    y_sample = np.atleast_2d(y_up[index])

    fl, lc = problem_l.evaluate(x_combo, return_values_of=['F','G'])
    print(fl)
    print(lc)

    # conduct local search with true evaluation
    localsearch_x, localsearch_f,_, n_fev, _, _, _, _ = \
        localsearch_on_trueEvaluation(xl, 100, 'lower', xu, problem_l, None, None, None, None)

    a = 0

    # let's plot the feasible and infeasible regions
    k = 1000
    test_data, _, _ = init_xy(k, problem_l, 0, **{'problem_type': 'bielvel'})
    test_combo = np.repeat(xu, k, axis=0)
    test_combo = np.hstack((test_combo, test_data))
    test_cc = problem_l.evaluate(test_combo, return_values_of=['G'])

    test_cc[np.abs(test_cc) < 1e-6] = 0
    test_c = copy.deepcopy(test_cc)

    test_c[test_c<=0] = 0
    test_c_cv = np.sum(test_c, axis=1)
    test_c_cv = test_c_cv.ravel()
    infeasible = np.nonzero(test_c_cv)
    a = np.linspace(0, k-1, k)
    feasible = np.setdiff1d(a, infeasible)
    infeasible = [int(b) for b in infeasible[0]]
    feasible = [int(b) for b in feasible]

    fig = plt.figure()
    # ------------------------------------------
    # plot upper level fu with exhausive search
    ax1 = fig.add_subplot(111)

    ax1.scatter(test_cc[infeasible, 0], test_cc[infeasible, 1], c='k')
    ax1.scatter(test_cc[feasible, 0], test_cc[feasible, 1], c='g')
    ax1.scatter(lc[0,0], lc[0, 1], marker='X',c='r')
    plt.show()
    a = 0

    '''
    eim_l = EI.EIM(problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=problem_l.xu,
                   lower_bound=problem_l.xl)

    matching_x, matching_f, n_fev_local, feasible_flag = \
        search_for_matching_otherlevel_x(xu,
                                         30,
                                         32,
                                         problem_l,
                                         'lower',
                                         eim_l,
                                         100,
                                         100,
                                         0, False, 'eim')

    testdata_y, xl, vio_list = upper_y_from_exghaustive_search(problem_l, problem_u, xu, 1000)


    print(testdata_y)
    a = 0
    '''



def SMD_invest_visualisation(problem_u, problem_l, seed):
    # this method is only a consequent
    # visualization method SMD_invest
    # only after the above method generates data
    # this method can be used
    from mpl_toolkits.mplot3d import Axes3D

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'



    # (1) load exhausive search results
    result_folder = working_folder + '\\' + folder + '\\real_function'

    savefile = result_folder + '\\' + problem + '_testdate_exh_seed_' + str(seed) + '.csv'
    testdata = np.loadtxt(savefile, delimiter=',')
    testdata_y = testdata[:, -1]

    # calculate fl from data
    complete_x = testdata[:, 0:-1]
    fl = problem_l.evaluate(complete_x, return_values_of=['F'])

    # read surrogate model saved results
    surdata = np.atleast_2d([0] * 7)
    for i in range(10):
        savefile = result_folder + '\\' + problem + '_testdate_sur_seed_' +\
                   str(seed) + '_' + str(i) + '.joblib'
        batch = joblib.load(savefile)
        surdata = np.vstack((surdata, batch))
    surdata = np.delete(surdata, 0, axis=0)
    fl_sur = surdata[:, -1]
    fu_sur = surdata[:, -2]

    xu_sur = surdata[:, 0: problem_u.n_levelvar]
    xl_sur = surdata[:, problem_u.n_levelvar: problem_u.n_var]


    # (2) create before local search
    # surrgoate approximation surface
    # with the above exhausive search data
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_krgmodels'
    krgmodel_save = result_folder + '\\krg_' + str(seed) + '.joblib'
    print(krgmodel_save)
    krg = joblib.load(krgmodel_save)
    krgmodel_save = result_folder + '\\krg_g_' + str(seed) + '.joblib'
    print(krgmodel_save)
    krg_g = joblib.load(krgmodel_save)

    # In order to plot the test data, we need the training data of krg
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    traindata_file = result_folder + '\\sampled_data_x_' + str(seed) + '.csv'
    print(traindata_file)
    x_both = np.loadtxt(traindata_file, delimiter=',')
    traindata_file = result_folder + '\\sampled_data_y_' + str(seed) + '.csv'
    print(traindata_file)
    y_up = np.loadtxt(traindata_file, delimiter=',')

    feasiflag_save = result_folder + '\\sampled_ll_feasi_' + str(seed) + '.joblib'
    # print(feasiflag_save)
    feasible_flag = joblib.load(feasiflag_save)

    # extract upper level variable
    train_size = 50
    train_x = np.atleast_2d(x_both[0: train_size, 0:problem_u.n_levelvar]).reshape(-1, problem_u.n_levelvar)
    train_y = np.atleast_2d(y_up[0: train_size]).reshape(-1, 1)
    test_x = np.atleast_2d(testdata[:, 0:problem_u.n_levelvar])
    pred_y = trained_model_prediction(krg[0], train_x, train_y, test_x, problem_u.n_levelvar)


    # re-identify local search starting point
    complete_x_u = np.atleast_2d(x_both[0: train_size, 0:problem_u.n_var])
    complete_y_u = np.atleast_2d(y_up[0: train_size]).reshape(-1, 1)
    complete_c_u = problem_u.evaluate(complete_x_u, return_values_of=["G"])
    best_xu, ankor_f = ankor_selection(complete_x_u, complete_y_u, complete_c_u, problem_u)
    ankor_xu = np.atleast_2d(best_xu)
    last_xu = np.atleast_2d(x_both[-2, 0:problem_u.n_levelvar])
    last_fu = np.atleast_2d(y_up[-2])

    fig = plt.figure()

    range_fu = []
    range_fu = np.append(range_fu, testdata_y)
    # range_fu = np.append(range_fu, fu_sur)
    z_max = np.max(range_fu) + 1
    z_min = np.min(range_fu) - 1

    range_fl = []
    range_fl = np.append(range_fu, fl)
    range_fl = np.append(range_fu, fl_sur)
    l_max = np.max(range_fu) + 1
    l_min = np.min(range_fu) - 1
    # the plot should have 4 subplots
    #

    fig = plt.figure(figsize=(20, 25))
    xu1_min = problem_u.xl[0]
    xu1_max = problem_u.xu[0]

    xu2_min = problem_u.xl[1]
    xu2_max = problem_u.xu[1]

    # ------------------------------------------
    # plot upper level fu with exhausive search
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_xlabel('xu1')
    ax1.set_xlim([xu1_min, xu1_max])
    ax1.set_ylabel('xu2')
    ax1.set_ylim([xu2_min, xu2_max])
    ax1.set_zlim([z_min, z_max])
    ax1.set_zlabel('fu')

    cm = plt.cm.get_cmap('RdYlBu')
    cl1 = ax1.scatter(testdata[:, 0], testdata[:, 1],
                      testdata_y.ravel(), c=testdata_y.ravel(),
                      cmap=cm, vmin=z_min, vmax=z_max)

    ax1.scatter(ankor_xu[:, 0], ankor_xu[:, 1], ankor_f.ravel(), marker='X', c='k')
    ax1.scatter(last_xu[:, 0], last_xu[:, 1], last_fu.ravel(), marker='X', c='g')

    ax1.set_title('fu with exhausive search')
    fig.colorbar(cl1, ax=ax1)


    # -------------------------------------------
    # plot lower level fl from exhausive search
    ax2 = fig.add_subplot(234, projection='3d')
    ax2.set_xlabel('xu1')
    ax2.set_xlim([xu1_min, xu1_max])
    ax2.set_ylabel('xu2')
    ax2.set_ylim([xu2_min, xu2_max])
    ax2.set_zlabel('fl')
    ax2.set_zlim([l_min, l_max])
    cl2 = ax2.scatter(testdata[:, 0], testdata[:, 1],
                      fl.ravel(), c=fl.ravel(),
                      cmap=cm, vmin=l_min, vmax=l_max)
    ax2.set_title('fl with exhausive search')

    fig.colorbar(cl2, ax=ax2)

    # -----------------------------------------
    # plot upper level fu from surrogate results
    ax3 = fig.add_subplot(232, projection='3d')
    ax3.set_xlabel('xu1')
    ax3.set_xlim([xu1_min, xu1_max])
    ax3.set_ylabel('xu2')
    ax3.set_ylim([xu2_min, xu2_max])
    ax3.set_zlim([z_min, z_max])
    cl3 = ax3.scatter(xu_sur[:, 0], xu_sur[:, 1],
                      fu_sur, c=fu_sur,
                      cmap=cm, vmin=z_min, vmax=z_max)
    ax3.scatter(ankor_xu[:, 0], ankor_xu[:, 1], ankor_f.ravel(), marker='X', c='k')
    ax3.set_title('fu with EGO hybrid search')
    fig.colorbar(cl3, ax=ax3)


    #-----------------------------------------
    # plot lower level fu from surrogate results
    ax4 = fig.add_subplot(235, projection='3d')
    ax4.set_xlabel('xu1')
    ax4.set_xlim([xu1_min, xu1_max])
    ax4.set_ylabel('xu2')
    ax4.set_ylim([xu2_min, xu2_max])
    ax4.set_zlim([l_min, l_max])
    cl4 = ax4.scatter(xu_sur[:, 0], xu_sur[:, 1],
                      fl_sur, c=fl_sur,
                      cmap=cm, vmin=l_min, vmax=l_max)
    ax4.set_title('fl with EGO hybrid search')
    fig.colorbar(cl4, ax=ax4)

    # plot surrogate prediction by the end of the upper EGO
    # ------------------------------------------
    # plot upper level fu with exhausive search
    ax5 = fig.add_subplot(233, projection='3d')
    ax5.set_xlabel('xu1')
    ax5.set_xlim([xu1_min, xu1_max])
    ax5.set_ylabel('xu2')
    ax5.set_ylim([xu2_min, xu2_max])
    # ax5.set_zlim([z_min, z_max])
    ax5.set_zlabel('fu')
    cm = plt.cm.get_cmap('RdYlBu')
    cl5 = ax5.scatter(test_x[:, 0], test_x[:, 1],
                   pred_y.ravel(), c=pred_y.ravel(),
                       cmap=cm)  #, vmin=l_min, vmax=l_max)
    ax5.scatter(ankor_xu[:, 0], ankor_xu[:, 1], ankor_f.ravel(), marker='X', c='k')
    ax5.scatter(last_xu[:, 0], last_xu[:, 1], last_fu.ravel(), marker='X', c='g')
    ax5.scatter(train_x[:, 0], train_x[:, 1], train_y.ravel(), marker='X', c='k')
    ax5.set_title('surrogate rebuild')
    # fig.colorbar(cl5, ax=ax5)
    start = '[' + "{:4.2f}".format(ankor_xu[:, 0][0]) + ', ' + "{:4.2f}".format(ankor_xu[:, 1][0]) + ']'
    ax5.text(0, 0, z_max, start)
    end = '[' + "{:4.2f}".format(last_xu[:, 0][0]) + ', ' + "{:4.2f}".format(last_xu[:, 1][0]) + ']'
    ax5.text(0, 0, z_max-50, end)



    plt.show()
    plot_save = result_folder + '\\' + problem + '_llsearch_vis_' + str(seed) + '.png'
    plt.savefig(plot_save,  format='png')

def SMD_investigate_kriging(problem_u, problem_l, seed):
    # This method aims at investigate the performance of kriging
    # (1) load train data and respond, isolate the kriging training, and predict 1000 sampled data
    # (2) same train data, but use exhaustive search to get f, rebuild 1000 points
    # (3) save the same train and respond data and move to comparison on matlab
    # In order to plot the test data, we need the training data of krg

    from mpl_toolkits.mplot3d import Axes3D
    vio_setting = 122  # 122 for SMD11
    test_xu, _, _ = init_xy(1000, problem_u, 0, **{'problem_type': 'bilevel'})

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'

    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    traindata_file = result_folder + '\\sampled_data_x_' + str(seed) + '.csv'
    print(traindata_file)
    x_both = np.loadtxt(traindata_file, delimiter=',')
    traindata_file = result_folder + '\\sampled_data_y_' + str(seed) + '.csv'
    print(traindata_file)
    y_up = np.loadtxt(traindata_file, delimiter=',')

    feasiflag_save = result_folder + '\\sampled_ll_feasi_' + str(seed) + '.joblib'
    # print(feasiflag_save)
    feasible_flag = joblib.load(feasiflag_save)

    # extract upper level variable
    train_size = 50

    train_x = np.atleast_2d(x_both[0: train_size, 0:problem_u.n_levelvar]).reshape(-1, problem_u.n_levelvar)
    train_y = np.atleast_2d(y_up[0: train_size]).reshape(-1, 1)
    train_c = None
    m = train_x.shape[0]
    print('before rebuild number of data: %d' % m)

    #-----fixed distance elimination
    # train_x, train_y, train_c = close_point_elimination(train_x, train_y, train_c)
    # n = train_x.shape[0]
    # print('after close elimination, number of data: %d' % n)

    # apply kriging
    # krg, krg_g = krg_training(train_x, train_y, None)
    #-------fixed distance elimination

    # use dynamically determined krg
    krg, krg_g, d_index = \
        dynamic_close_point_elimination(train_x, train_y, train_c, [1, 2, 3])

    print(d_index)

    train_x, train_y, train_c = close_point_elimination(train_x, train_y, train_c, d_index + 1)
    pred_y = trained_model_prediction(krg[0], train_x, train_y, test_xu, problem_u.n_levelvar)


    # (2) find exhaustive search xl for train_x
    #     then evaluate fu
    #     use these xu-fu pair to build krg
    test_fu, xl, vio_list = upper_y_from_exghaustive_search(problem_l, problem_u, train_x, vio_setting)

    krg_exh, krg_g_exh = krg_training(train_x, test_fu, None)
    pred_y_exh = trained_model_prediction(krg_exh[0], train_x, test_fu, test_xu, problem_u.n_levelvar)

    # (3) visualisation
    fig = plt.figure(figsize=(15, 20))
    xu1_min = problem_u.xl[0]
    xu1_max = problem_u.xu[0]

    xu2_min = problem_u.xl[1]
    xu2_max = problem_u.xu[1]
    
    # ------------------------------------------
    # plot upper level fu with exhausive search
    # prepare a 2d plot of infeasible and feasible
    # warning: this only works for SMD10!!!
    xu1_grid = np.linspace(problem_u.xl[0], problem_u.xu[0], 100)
    xu2_grid = np.linspace(problem_u.xl[1], problem_u.xu[1], 100)
    xu1_mesh, xu2_mesh = np.meshgrid(xu1_grid, xu2_grid)
    cons = test_SMD10_contraints_visual(xu1_mesh, xu2_mesh)
    feasi_index = np.argwhere(cons > 0)
    feasi_index_1 = np.atleast_2d(feasi_index).reshape(-1, 2)[:, 0]
    feasi_index_2 = np.atleast_2d(feasi_index).reshape(-1, 2)[:, 1]
    xu1_c = xu1_mesh[feasi_index_1, feasi_index_2]
    xu2_c = xu2_mesh[feasi_index_1, feasi_index_2]


    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('xu1')
    ax1.set_xlim([xu1_min, xu1_max])
    ax1.set_ylabel('xu2')
    ax1.set_ylim([xu2_min, xu2_max])
    # ax1.set_zlim([z_min, z_max])
    ax1.set_zlabel('fu')
    cm = plt.cm.get_cmap('rainbow')
    cl1 = ax1.scatter(test_xu[:, 0], test_xu[:, 1],
                      pred_y_exh.ravel(), c=pred_y_exh.ravel(),
                      cmap=cm)
    ax1.scatter(train_x[:, 0], train_x[:, 1], test_fu.ravel(), c='k', marker ='X')

    zlim = ax1.get_zlim()
    ax1.scatter(xu1_c, xu2_c, zs=zlim[0], zdir='z')

    ax1.set_title('kriging with exhausive search')
    fig.colorbar(cl1, ax=ax1)
    

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('xu1')
    ax2.set_xlim([xu1_min, xu1_max])
    ax2.set_ylabel('xu2')
    ax2.set_ylim([xu2_min, xu2_max])
    # ax1.set_zlim([z_min, z_max])
    ax2.set_zlabel('fu')
    cm = plt.cm.get_cmap('rainbow')
    cl2 = ax2.scatter(test_xu[:, 0], test_xu[:, 1],
                      pred_y.ravel(), c=pred_y.ravel(),
                      cmap=cm)

    ax2.scatter(train_x[:, 0], train_x[:, 1], train_y.ravel(), c='k', marker='X')
    zlim = ax2.get_zlim()
    ax2.scatter(train_x[:, 0], train_x[:, 1], zs=zlim[0], zdir='z', c='r', marker='X')

    t = 'kriging rebuild with EGO search, ' + str(d_index + 1) + ' closeness selected'
    ax2.set_title(t)
    fig.colorbar(cl2, ax=ax2)
    plt.show()

    # (3) save train_x, train_y into csv file to study performance in matlab
    save_x = 'train_x.csv'
    save_y = 'train_y.csv'
    save_test = 'test_x.csv'
    np.savetxt(save_x, train_x, delimiter=',')
    np.savetxt(save_y, train_y, delimiter=',')
    np.savetxt(save_test, test_xu, delimiter=',')

def SMD_distance_measure_compare(problem_u):
    print('Not done')
    from mpl_toolkits.mplot3d import Axes3D
    vio_setting = 122  # 122 for SMD11

    test_xu, _, _ = init_xy(1000, problem_u, 0, **{'problem_type': 'bilevel'})

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'

    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    traindata_file = result_folder + '\\sampled_data_x_' + str(seed) + '.csv'
    print(traindata_file)
    x_both = np.loadtxt(traindata_file, delimiter=',')
    traindata_file = result_folder + '\\sampled_data_y_' + str(seed) + '.csv'
    print(traindata_file)
    y_up = np.loadtxt(traindata_file, delimiter=',')

    feasiflag_save = result_folder + '\\sampled_ll_feasi_' + str(seed) + '.joblib'
    # print(feasiflag_save)
    feasible_flag = joblib.load(feasiflag_save)

    # extract upper level variable
    train_size = 50

    train_x = np.atleast_2d(x_both[0: train_size, 0:problem_u.n_levelvar]).reshape(-1, problem_u.n_levelvar)
    train_y = np.atleast_2d(y_up[0: train_size]).reshape(-1, 1)
    train_c = None
    m = train_x.shape[0]
    print('before rebuild number of data: %d' % m)

    fig = plt.figure(figsize=(15, 20))
    xu1_min = problem_u.xl[0]
    xu1_max = problem_u.xu[0]

    xu2_min = problem_u.xl[1]
    xu2_max = problem_u.xu[1]



    train_x_2, train_y_2, train_c_2 = close_point_elimination(train_x, train_y, train_c, 2)
    n = train_x_2.shape[0]
    print('after close elimination, number of data: %d' % n)
    krg2, krg_g2 = krg_training(train_x_2, train_y_2, None)
    pred_y2= trained_model_prediction(krg2[0], train_x_2, train_y_2, test_xu, problem_u.n_levelvar)



    train_x_1, train_y_1, train_c_1 = close_point_elimination(train_x, train_y, train_c, 1)
    n = train_x_1.shape[0]
    print('after close elimination, number of data: %d' % n)
    krg1, krg_g1 = krg_training(train_x_1, train_y_1, None)
    pred_y1 = trained_model_prediction(krg1[0], train_x_1, train_y_1, test_xu, problem_u.n_levelvar)

    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.set_xlabel('xu1')
    ax2.set_xlim([xu1_min, xu1_max])
    ax2.set_ylabel('xu2')
    ax2.set_ylim([xu2_min, xu2_max])
    # ax1.set_zlim([z_min, z_max])
    ax2.set_zlabel('fu')
    cm = plt.cm.get_cmap('rainbow')
    cl2 = ax2.scatter(test_xu[:, 0], test_xu[:, 1],
                      pred_y2.ravel(), c=pred_y2.ravel(),
                      cmap=cm)

    ax2.scatter(train_x_2[:, 0], train_x_2[:, 1], train_y_2.ravel(), c='k', marker='X')
    ax2.set_title('kriging rebuild with EGO search, 0.00 closeness check')
    fig.colorbar(cl2, ax=ax2)

    ax1 = fig.add_subplot(122, projection='3d')
    ax1.set_xlabel('xu1')
    ax1.set_xlim([xu1_min, xu1_max])
    ax1.set_ylabel('xu2')
    ax1.set_ylim([xu2_min, xu2_max])
    # ax1.set_zlim([z_min, z_max])
    ax1.set_zlabel('fu')
    cm = plt.cm.get_cmap('rainbow')
    cl1 = ax1.scatter(test_xu[:, 0], test_xu[:, 1],
                      pred_y1.ravel(), c=pred_y1.ravel(),
                      cmap=cm)

    ax1.scatter(train_x_1[:, 0], train_x_1[:, 1], train_y_1.ravel(), c='k', marker='X')
    ax1.set_title('kriging rebuild with EGO search, 0.0 closeness check')
    fig.colorbar(cl1, ax=ax1)
    plt.show()

def close_point_elimination(train_x_orig, train_y_orig, train_c_orig, measure):


    train_x_orig = check_array(train_x_orig)
    train_y_orig = check_array(train_y_orig)


    if train_c_orig is not None:
        train_c_orig = check_array(train_c_orig)
        train_c = copy.deepcopy(train_c_orig)
    else:
        train_c = None

    train_x = copy.deepcopy(train_x_orig)
    train_y = copy.deepcopy(train_y_orig)

    # train_x = train_x_orig
    # train_y = train_y_orig

    # train_c = train_c_orig

    norm_x = normalization_with_self(train_x)
    norm_x, id = np.unique(np.around(norm_x, measure), axis=0, return_index=True)
    z = len(id)
    # print('after elimiation train size becomes: %d' %z)
    # n = train_x_orig.shape[0]
    # m = len(id)
    # print('before closeness elimination: %d' % n)
    # print('after closeness elimination: %d' % m)

    train_x = np.atleast_2d(train_x[id, :])
    train_y = np.atleast_2d(train_y[id, :])
    if train_c is not None:
        train_c = np.atleast_2d(train_c[id, :])
    return train_x, train_y, train_c


def krg_training(train_x, train_y, train_c):
    train_x_norm, x_mean, x_std = norm_by_zscore(train_x)
    train_y_norm, y_mean, y_std = norm_by_zscore(train_y)

    train_x_norm = np.atleast_2d(train_x_norm)
    train_y_norm = np.atleast_2d(train_y_norm)

    if train_c is not None:
        train_c_norm, c_mean, c_std = norm_by_zscore(train_c)
        train_c_norm = np.atleast_2d(train_c_norm)
    else:
        train_c_norm = None

    krg, krg_g = cross_val_krg(train_x_norm, train_y_norm, train_c_norm, False)
    return krg, krg_g

def saveEGOtraining(complete_xu, complete_yu, folder, problem_u, feasible_check, seed):

    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + folder + '\\' + problem + '_sampleddata'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    egodata_save = result_folder + '\\sampled_data_x_'+str(seed)+'.csv'
    np.savetxt(egodata_save, complete_xu, delimiter=',')

    egodata_save = result_folder + '\\sampled_data_y_'+str(seed)+'.csv'
    np.savetxt(egodata_save, complete_yu, delimiter=',')

    feasi_save = result_folder + '\\sampled_ll_feasi_' + str(seed) + '.joblib'
    joblib.dump(feasible_check, feasi_save)



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

    # seedlist = [0, 5, 8, 6, 5, 4, 1, 9]  # test only

    seed_i = 0
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
        #------- move back for indent correction, when uncomment the above line
            # seed_index = seedlist[seed_i]  # test only
            # saveName = result_folder + '\\accuracy_before_evaluation' + str(seed_index) + '.csv'
            saveName = result_folder + '\\accuracy_' + str(seed_index) + '.csv'
            data = np.loadtxt(saveName, delimiter=',')
            accuracy_data = np.append(accuracy_data, data)
            seed_i = seed_i + 1  # test only
        #-------------------------------------------------------------------------

        accuracy_data = np.atleast_2d(accuracy_data).reshape(-1, 2)

        # select median with ul and corresponding ll
        ul_accuracy = np.array(accuracy_data[:, 0])
        seed_median = np.argsort(ul_accuracy)[int(seedmax/2)]
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

    # seedlist = [0, 5, 8, 6, 5, 4, 1, 9]  # test only
    seed_i = 0

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
            # ------this section to mark where to indent when uncomment the following 'for' line
            for seed_index in range(max_seed):
            # seed_index = seedlist[seed_i]
                saveName = result_folder + '\\ll_nfev' + str(seed_index) + '.csv'
                data = np.loadtxt(saveName, delimiter=',')
                n_fev = np.append(n_fev, data)
            #----------------------------------------------

            mean_nfev = np.median(n_fev)
            mean_smds = np.append(mean_smds, mean_nfev)

            seed_i = seed_i + 1

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
    # standlone method to plot smd3 upper level countour
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

def uplevel_localsearch_visual(train_x, train_y, train_c, n_before_ls,
                               krg, krg_g, seed, folder, problem_u):
    # this function plots surrogate's samples
    # upper level local search starting point
    # upper level local search end point (this can be a colorschemed plot)
    # the plot can contain 3 parts, real function (1), surrogate (2), and (3) local search

    # (1) load test data for plotting real function
    working_folder = os.getcwd()
    result_folder = working_folder + '\\' + folder + '\\real_function'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    problem = problem_u.name()[0:-2]
    savefile = result_folder + '\\' + problem + 'testdate_gen_seed_' + str(seed) + '.csv'
    testdata = np.loadtxt(savefile, delimiter=',')
    test_x = testdata[:, 0]  # only process one variable problem
    test_y = testdata[:, -1]  # only process one variable problem

    # real function also needs plot infeasibility
    n_c = problem_u.n_constr
    if n_c > 0:
        both_x = np.atleast_2d(testdata[:, 0:-1]).reshape(-1, problem_u.n_var)
        test_c = problem_u.evaluate(both_x, return_values_of=['G'])
        test_y = np.atleast_2d(test_y).reshape(-1, 1)
        test_x = np.atleast_2d(test_x).reshape(-1, 1)

        test_inf = get_infeasible_subset(test_x, test_y, test_c)


    # (2) prepare data for plotting surrogate
    # use test_x for creating surrogate for plot
    # make sure the train_x, train_y from EGO main process has no extra sample to ruin the krg prediction
    before_ls_x = np.atleast_2d(train_x[0: n_before_ls, :]).reshape(-1, problem_u.n_var)
    before_ls_y = np.atleast_2d(train_y[0: n_before_ls, :]).reshape(-1, problem_u.n_obj)
    if n_c > 0:
        before_ls_c = np.atleast_2d(train_c[0: n_before_ls, :]).reshape(-1, problem_u.n_constr)

    train_x_of_surrogate = np.atleast_2d(before_ls_x[:, 0:problem_u.n_levelvar])
    train_y_of_surrogate = np.atleast_2d(before_ls_y[:, 0:problem_u.n_obj])
    if n_c > 0:
        train_c_of_surrogate = np.atleast_2d(before_ls_c[:, 0:problem_u.n_constr])

    pred_y = trained_model_prediction(krg[0], train_x_of_surrogate, train_y_of_surrogate, test_x)
    pred_y = np.atleast_2d(pred_y)

    # consider the problem with constraint
    # mark those in test that are violation of contraints

    cons_y = []
    if n_c > 0:
        for i in range(n_c):
            train_c_surr_column = np.atleast_2d(train_c_of_surrogate[:, i]).reshape(-1, 1)  # one constraint
            cons_c = trained_model_prediction(krg_g[i], train_x_of_surrogate, train_c_surr_column, test_x)
            cons_y = np.append(cons_y, cons_c)
        cons_y = np.atleast_2d(cons_y).reshape(-1, n_c, order='F')
        test_x = np.atleast_2d(test_x).reshape(-1, 1)
        infeasi_pred_pair = get_infeasible_subset(test_x, pred_y, cons_y)

    # (3) prepare data to plot local search plot
    # identify local search starting point
    if n_c == 0:
        before_ls_c = None

    # ankor returned level variable
    selected_x, selected_f = ankor_selection(before_ls_x, before_ls_y, before_ls_c, problem_u)

    after_ls_x = np.atleast_2d(train_x[n_before_ls:, 0:problem_u.n_levelvar])  # can there be a problem for one instance?
    after_ls_y = np.atleast_2d(train_y[n_before_ls:, 0:problem_u.n_levelvar])
    n = after_ls_x.shape[0]
    order = np.linspace(0, n, n)

    # time to plot
    # plt.ion()
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(131)
    ax1.set_xlabel('design variable')
    ax1.set_ylabel('f value')
    ax1.scatter(test_x.ravel(), test_y, c='r')
    ax1.set_title('function got from exghausive search')

    # plot surrogate
    ax2 = fig.add_subplot(132)
    ax2.set_xlabel('design variable')
    ax2.set_ylabel('surrogate predicted f value')
    ax2.scatter(test_x.ravel(), pred_y.ravel(), c='g')
    ax2.scatter(train_x_of_surrogate, train_y_of_surrogate, marker='x', c='r')
    if n_c > 0:
        ax1.scatter(test_inf[:, 0], test_inf[:, 1], s=1, c='k')
        ax2.scatter(infeasi_pred_pair[:, 0], infeasi_pred_pair[:, 1], s=1, c='k')

    ax2.set_title('surrogates approximation')

    # plot local search
    cm3 = plt.cm.get_cmap('RdYlBu')
    ax3 = fig.add_subplot(133)
    ax3.set_xlabel('design variable')
    ax3.set_ylabel('local search based on current training data')
    cs3 = ax3.scatter(after_ls_x.ravel(), after_ls_y.ravel(), c=order, cmap=cm3)

    ax3.scatter(selected_x[0], selected_f[0], c='k', marker='X')
    ax2.scatter(selected_x[0], selected_f[0], c='k', marker='X') # plot to 2nd plot
    if n_c > 0:
        ax2.legend(['surrogates prediction', 'training samples', 'infeasible', 'ls start pick'])
    else:
        ax2.legend(['surrogate prediction', 'training samples', 'ls start pick'])
    ax3.scatter(after_ls_x.ravel()[-1], after_ls_y.ravel()[-1], c='g', marker='X')
    fig.colorbar(cs3, ax=ax3)
    ax3.legend(['ls progress colormap', 'ls start', 'ls end'])
    ax3.set_title('upper level local search progress')

    # scale to the same y scale
    y_range = []
    y_range = np.append(y_range, pred_y)  # y_range = pred_y + train_y
    y_range = np.append(y_range, train_y)
    y_max = np.max(y_range) + 0.1
    y_min = np.min(y_range) - 0.1
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax3.set_ylim(y_min, y_max)

    x_min = problem_u.xl
    x_max = problem_u.xu
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax3.set_xlim(x_min, x_max)


    result_folder = working_folder + '\\' + folder + '\\upper_ls_plot'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    savefile = result_folder + '\\' + problem + '_upper_ls_seed_' + str(seed) + '.png'
    plt.savefig(savefile)

    # plt.show()

def test_if(a, b):
    a = (True, False) if a > b else (False, True)
    return zip(a)

def test_SMD10_contraints_visual(gridxu1, gridxu2):
    # if feasible cons = 1
    # if not feasible cons = -1
    cons1 = gridxu1 - gridxu2 ** 3
    cons2 = gridxu2 - gridxu1 ** 3

    cons1[cons1 >= 0] = 1
    cons2[cons2 >= 0] = 1
    cons1[cons1 < 0] = -2
    cons2[cons2 < 0] = -2
    cons = cons1 +  cons2

    cons[cons<0] = -1
    cons[cons>0] = 1
    return cons

def test_cross_val(problem_l):
    xu = np.atleast_2d([-1.2, -1.2])
    level = 'lower'

    np.set_printoptions(precision=16)
    # for testing SMD problem;
    D = problem_l.n_levelvar
    n_samples = 11 * D - 1

    train_x, train_y, cons_y = init_xy(n_samples, problem_l, 0,
                                       **{'problem_type': 'bilevel'})
    num_l = train_x.shape[0]
    # print(train_x[0, :])
    x_other = np.atleast_2d(xu)
    xother_expand = np.repeat(x_other, num_l, axis=0)
    if level == 'lower':
        complete_x = np.hstack((xother_expand, train_x))
    else:
        complete_x = np.hstack((train_x, xother_expand))

    if problem_l.n_constr > 0:
        # print("constraint settings")
        complete_y, complete_c = problem_l.evaluate(complete_x, return_values_of=["F", "G"])

    else:
        complete_y = problem_l.evaluate(complete_x, return_values_of=["F"])
        complete_c = None

    # cross_val_krg(train_x, complete_y, complete_c, True)
    a, b, c, d = dynamic_close_point_elimination(train_x, complete_y, complete_c, [1,2,3])




if __name__ == "__main__":
    import matlab.engine
    eng = matlab.engine.start_matlab()

    tf = eng.isprime(37)
    print(tf)

    seed = 0
    # rebuild_surrogate_and_plot()

    # problems_json = 'p/bi_problems'
    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    np.random.seed(seed)
    x = np.pi/2 - 1e-8
    print(x)
    print(1.5608265570618167)

    # x = np.atleast_2d([-4.05, -4.99, 1.56])
    # bu = [10, 10, 1.57]
    # bl = [-5, -5, -1.57]
    # x = boundary_closeness_check(x, bu, bl)

    '''
    x = np.arange(9) - 4
    b = x.reshape((-1, 3))
    c = np.linalg.norm(b,ord=1, axis=0)
    print(b)
    print(c)

    d = b/c
    print(d)
    '''

    problem_u = eval(target_problems[2])
    problem_l = eval(target_problems[3])
    # test_cross_val(problem_l)
    np.set_printoptions(precision=16)
    xu = np.atleast_2d([-1.2136516906077681, -1.228916217819073])
    xl = np.atleast_2d([-1.8197992672355463, -1.7274253188436908, 1.5608265570618167])
    # fu, gu = problem_u.evaluate(np.hstack((xu, xl)), return_values_of=['F', 'G'])
    # g = problem_l.evaluate(np.hstack((xu, xl)), return_values_of=['G'])
    # print(f)
    # print(g)



    # xl = np.atleast_2d([-3.977216235870402,  -2.533006647477677,  -0.9727335525247666])
    localsearch_x, localsearch_f, success, n_fev, _, _, _, _ = \
         localsearch_on_trueEvaluation(xl, 100, 'lower', xu, problem_l, 'slsqp', None, None, None, None)
    print(success)
    print(localsearch_x)
    g = problem_l.evaluate(np.hstack((xu, np.atleast_2d(localsearch_x))), return_values_of=['G'])
    print(g)



    '''
    best_xu_sofar = xu
    complete_x_u = np.atleast_2d(np.hstack((xu, xl)))
    complete_y_u = np.atleast_2d(fu)
    complete_c_u = np.atleast_2d(gu)

    eim_l = EI.EIM(problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=problem_l.xu,
                   lower_bound=problem_l.xl)
    feasible_check = [True]

    bi_para = \
        {'search_iter': 30,
         'n_samples': 30,
         'problem': problem_l,
         'level': 'lower',
         'eim': eim_l,
         'eim_pop': 100,
         'eim_gen': 100,
         'seed_index': 0,
         'enable_crossvalidation': False,
         'method_selection': 'eim'}

    t1 = time.time()
    new_xu, new_fu, success, n_fev_local, feasible_check, complete_x_u, complete_y_u, complete_c_u = \
        localsearch_on_trueEvaluation(best_xu_sofar,
                                      10,
                                      'upper',
                                      None,
                                      problem_u,
                                      complete_x_u,
                                      complete_y_u,
                                      complete_c_u,
                                      feasible_check,
                                      **bi_para)
    t2 = time.time()
    elapse = (t2 - t1)/60
    print(elapse)

    # test on bilevel local search
    '''

    '''
    delta1 = [1e-8, 0, 0]
    delta2 =[0, 1e-8, 0]
    delta3 = [0, 0, 1e-8]

    fl1 = problem_l.evaluate(np.hstack((xu, xl+delta1)), return_values_of=['F'])
    fl2 = problem_l.evaluate(np.hstack((xu, xl + delta2)), return_values_of=['F'])
    fl3 = problem_l.evaluate(np.hstack((xu, xl + delta3)), return_values_of=['F'])
    print(f)
    print(fl1)
    print(fl2)
    print(fl3)
    print((fl1-f)/1e-8)
    print((fl2-f)/1e-8)
    print((fl3-f)/1e-8)
    '''




    # xu = np.atleast_2d([1., 1.])
    # xl = np.atleast_2d([1., 1., np.pi/4.0])
    # x = np.hstack((xu, xl))
    # fu, cu = problem_u.evaluate(x, return_values_of=['F','G'])
    # fl, cl = problem_l.evaluate(x, return_values_of = ['F', 'G'])

    # SMD_invest_retrieve_vio(problem_u, 'bi_output', 0)
    #SMD_invest(problem_u, problem_l, None, 0)

    # SMD_invest_visualisation(problem_u, problem_l, seed)
    # SMD_investigate_kriging(problem_u, problem_l, seed)
    # SMD_distance_measure_compare(problem_u)

    '''
    xu = np.atleast_2d([-3.85516901709, 9.597879966999528])
    eim_l = EI.EIM(problem_l.n_levelvar, n_obj=1, n_constr=0,
                   upper_bound=problem_l.xu,
                   lower_bound=problem_l.xl)
    matching_x, matching_f, n_fev_local, feasible_flag = \
        search_for_matching_otherlevel_x(xu, 20, 30, problem_l, 'lower', eim_l, 100, 100, seed, False, 'eim')

    print(matching_f)
    
    xu1_grid = np.linspace(-5, 10, 100)
    xu2_grid = np.linspace(-5, 10, 100)
    xu1_mesh, xu2_mesh = np.meshgrid(xu1_grid, xu2_grid)
    cons = test_SMD10_contraints_visual(xu1_mesh, xu2_mesh)

    feasi_index = np.argwhere(cons > 0)
    infeasi_index = np.argwhere(cons < 0)

    feasi_index_1 = np.atleast_2d(feasi_index).reshape(-1, 2)[:, 0]
    feasi_index_2 = np.atleast_2d(feasi_index).reshape(-1, 2)[:, 1]

    infeasi_index_1 = np.atleast_2d(infeasi_index).reshape(-1, 2)[:, 0]
    infeasi_index_2 = np.atleast_2d(infeasi_index).reshape(-1, 2)[:, 1]

    fc_xu1 = xu1_mesh[feasi_index_1, feasi_index_2]
    fc_xu2 = xu2_mesh[feasi_index_1, feasi_index_2]


    a = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(fc_xu1, fc_xu2)
    ax1.set_xlim(-5, 10)
    ax1.set_ylim(-5, 10)
    plt.show()
    '''



    # problem_u = eval("SMD.SMD10_F(1,1,1)")
    # problem_l = eval("SMD.SMD10_f(1,1,1)")
    # invest_SMD_one_instance(problem_u, problem_l, 22, 0)

    '''
    working_folder = os.getcwd()
    result_folder = working_folder + '\\bi_output\\real_function'
    filename = result_folder + '\\SMD9_testdate_sur_seed_0_0.joblib'
    section = joblib.load(filename)
    a = 0
    
    # ----parallel data generation-----
    problem = problem_u.name()[0:-2]
    working_folder = os.getcwd()
    folder = 'bi_output'

    # save these test data
    result_folder = working_folder + '\\' + folder + '\\real_function'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    vio_setting = SMD_invest_retrieve_vio(problem_u, 'bi_output', 0)

    # create test data from variable bounds
    # testdata = np.linspace(problem_u.xl, problem_u.xu, 1000)
    testdata, _, _ = init_xy(1000, problem_u, 0, **{'problem_type': 'bilevel'})

    # surrogate_react_2_testdata(0, testdata, problem_u, problem_l, result_folder, seed)

    args = test_data_parall(testdata, problem_u, problem_l, result_folder, seed, vio_setting)
    # args = paral_args_temp(target_problems, seed_max, False, methods_ops, alg_settings)
    num_workers = 5
    pool = mp.Pool(processes=num_workers)
    pool.starmap(surrogate_react_2_testdata, ([arg for arg in args]))
    # ----parallel data generation-----
    '''




    # EGO_rebuildplot(problem_u, 'bi_output')

    # in general post process
    # ------------ result process--------------
    # problems = target_problems
    # folder = hyp['alg_settings']['folder']
    # results_process_bestf(problems, 'eim', 5, folder)
    # combine_fev(problems, 'eim', 5)
    # results_process_before_after(problems, 'eim', 'bi_output', 'accuracy', 29)
    # --------------result process ------------

