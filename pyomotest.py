from __future__ import division, print_function, unicode_literals


import numpy as np
import json
from surrogate_problems import SMD
import pygmo as pg
from sklearn.utils.validation import check_array

import numpy as np



def obj(x):
    xu1 = -1.2136516906077681
    xu2 = -1.228916217819073

    xl1 = x[0:2]
    xl2 = x[2]

    f = xu1 ** 2 + (xl1[0] - 2) ** 2 + (xl1[1] - 2) ** 2 + (xu2 - np.tan(xl2)) ** 2
    return f
def cons(x):
    xu1 = -1.2136516906077681
    xu2 = -1.228916217819073

    xl1 = x[0:2]
    xl2 = x[2]

    g1 = xl1[0] - xl1[1] ** 3
    g2 = xl1[1] - xl1[0] ** 3
    return [-g1, -g2]
class myproblem():

    def fitness(self, x):
        xu1 = -1.2136516906077681
        xu2 = -1.228916217819073

        xl1 = x[0:2]
        xl2 = x[2]

        f = xu1 ** 2 + (xl1[0]-2)**2 + (xl1[1] -2)**2 + (xu2-np.tan(xl2))**2

        g1 = xl1[0] - xl1[1] ** 3
        g2 = xl1[1] - xl1[0] ** 3
        return [f, -g1, -g2]

    def get_bounds(self):
        return ([-5, -5, -np.pi/2], [10, 10, np.pi/2])

    def get_nec(self):
        return 0

    def get_nic(self):
        return 2
class testProblem(object):
    def __init__(self):
        pass

    def objective(self, x):
        xu1 = -1.2136516906077681
        xu2 = -1.228916217819073

        xl1 = x[0:2]
        xl2 = x[2]
        f = xu1 ** 2 + (xl1[0] - 2) ** 2 + (xl1[1] - 2) ** 2 + (xu2 - np.tan(xl2)) ** 2
        return np.array(f)

    def constraints(self, x):
        xl1 = x[0:2]
        xl2 = x[2]
        g1 = xl1[0] - xl1[1] ** 3
        g2 = xl1[1] - xl1[0] ** 3
        return np.array((-g1, -g2))

    def gradient(self, x):
        eps = 1e-8
        f0 = self.objective(x)
        jac = np.zeros(len(x))
        dx = np.zeros(len(x))

        for i in range(len(x)):
            dx[i] = eps
            jac[i] = ( self.objective(x+dx)-f0)/eps
            dx[i] = 0.0

        return np.array(jac)


    def jacobian(self, x):
        eps = 1e-8
        g0 = self.constraints(x)
        jac = np.zeros([len(x), len(g0)])
        dx = np.zeros(len(x))

        for i in range(len(x)):
            dx[i] = eps
            jac[i] = (self.constraints(x + dx) - g0) / eps
            dx[i] = 0.0
        jac = jac.transpose().flatten()
        return np.array(jac)
class hs071(object):
    def __init__(self):
        pass

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]),
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return np.concatenate((np.prod(x) / x, 2*x))

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #

        return np.nonzero(np.tril(np.ones((4, 4))))

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        row, col = self.hessianstructure()

        return H[row, col]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

'''
def main():
    #
    # Define the problem
    #
    x0 = [1.0, 5.0, 5.0, 1.0]

    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]

    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]

    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=hs071(),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'second-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)

    #
    # Scale the problem (Just for demonstration purposes)
    #
    nlp.setProblemScaling(
        obj_scaling=2,
        x_scaling=[1, 1, 1, 1]
        )
    nlp.addOption('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))

'''

if __name__ == "__main__":

    from scipy.optimize import rosen, rosen_der
    from ipopt import minimize_ipopt

    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = minimize_ipopt(rosen, x0, jac=rosen_der)
    print(res)



    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    seed = 0
    np.random.seed(seed)

    problem_u = eval(target_problems[2])
    problem_l = eval(target_problems[3])
    # test_cross_val(problem_l)
    np.set_printoptions(precision=16)
    xu = np.atleast_2d([-1.2136516906077681, -1.228916217819073])
    xl = np.atleast_2d([-1.8197992672355463, -1.7274253188436908, 1.5608265570618167])




    '''
    # main()
    x0 = [-1.8197992672355463, -1.7274253188436908, 1.5608265570618167]
    lb = [-5, -5, -np.pi/2]
    ub = [10, 10, np.pi/2]

    cl = [-np.inf, -np.inf]
    cu = [0, 0]

    nlp = ipopt.problem(
        n=len(x0),
        m=len(cl),
        problem_obj=testProblem(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    '''

    '''
    nlp.setProblemScaling(
        obj_scaling=2,
        x_scaling=[1, 1, 1]
    )
    nlp.addOption('nlp_scaling_method', 'user-scaling')
    '''
    # nlp.addOption('mu_strategy', 'adaptive')
    # nlp.addOption('tol', 1e-7)

    # x, info = nlp.solve(x0)
    # print("Solution of the primal variables: x=%s\n" % repr(x))
    # print("Objective=%s\n" % repr(info['obj_val']))
    # print(x)
    # print(info)
    # g = testProblem().constraints(x)
    # print(g)




    '''

    # pg.test.run_test_suite()
    problems_json = 'p/bi_problems_test'
    with open(problems_json, 'r') as data_file:
        hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    seed = 0
    np.random.seed(seed)

    problem_u = eval(target_problems[2])
    problem_l = eval(target_problems[3])
    # test_cross_val(problem_l)
    np.set_printoptions(precision=16)
    xu = np.atleast_2d([-1.2136516906077681, -1.228916217819073])
    xl = np.atleast_2d([-1.8197992672355463, -1.7274253188436908, 1.5608265570618167])

    '''
    '''
    # problem = pg.problem(myproblem())
    # print(problem)
    nl = pg.nlopt('slsqp')
    ip = pg.ipopt()
    ip.set_numeric_option("tol", 1E-9)
    # algo = pg.algorithm(uda = pg.mbh(pg.nlopt("slsqp"), stop = 20, perturb = .1))
    algo = pg.algorithm(ip)
    # algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
    algo.set_verbosity(1)
    pop = pg.population(prob=myproblem())
    pop.push_back(x = [-1.8197992672355463, -1.7274253188436908, 1.5608265570618167])
    print(pop)


    pop.problem.c_tol=[1e-10]*2
    pop=algo.evolve(pop)
    print(pop.problem.get_fevals())
    print(pop.champion_f[0])
    p = myproblem()
    print(p.fitness(pop.champion_x))
        # print(pop.get_fevals())



    bounds = scipy.optimize.Bounds(lb=[-5, -5, -np.pi/2], ub=[10, 10, np.pi/2])
    optimization_res = scipy.optimize.minimize(obj, x0, method='slsqp',
                                               options={'maxiter': 100, 'disp': 3, 'iprint': 3},
                                               constraints={'type': 'ineq', 'fun': cons},
                                               bounds=bounds,
                                               tol=1e-10)
    '''

