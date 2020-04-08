import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg



class f(Problem):
    def __init__(self):
        self.n_var = 1
        self.n_constr = 0
        self.n_obj = 1
        self.xl = anp.array([0])
        self.xu = anp.array([1])
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)



    def _evaluate(self, x, out, *args, **kwargs):
        y = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        out["F"] = y

class lower_level_test1(Problem):
    def __init__(self):
        self.n_var = 3
        self.n_constr = 0
        self.n_obj = 1

        xl1_u = [10] * 2
        xl1_l = [-5] * 2

        xl2_u = [np.pi / 2] * 1
        xl2_l = [-np.pi / 2] * 1

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)



    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        xl1 = np.atleast_2d(x[:, 0:2]).reshape(n, -1)
        xl2 = np.atleast_2d(x[:, 2]).reshape(n, -1)

        f1 = 0
        f2 = np.sum(xl1 ** 2, axis=1)
        f3 = np.sum((0 - np.tan(xl2)) ** 2, axis=1)

        out["F"] = f1 + f2 + f3

class quadratic(Problem):
    def __init__(self):
        self.n_var = 2
        self.n_constr = 0
        self.n_obj = 1

        xl1_u = [10] * 1
        xl1_l = [-5] * 1

        xl2_u = [np.pi / 2] * 1
        xl2_l = [-np.pi / 2] * 1

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        out["F"] = np.sum(x**2, axis=1)

class lower_level_test1(Problem):
    def __init__(self):
        self.n_var = 3
        self.n_constr = 0
        self.n_obj = 1

        xl1_u = [10] * 2
        xl1_l = [-5] * 2

        xl2_u = [np.pi / 2] * 1
        xl2_l = [-np.pi / 2] * 1

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)



    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        xl1 = np.atleast_2d(x[:, 0:2]).reshape(n, -1)
        xl2 = np.atleast_2d(x[:, 2]).reshape(n, -1)

        f1 = 0
        f2 = np.sum(xl1 ** 2, axis=1)
        f3 = np.sum((0 - np.tan(xl2)) ** 2, axis=1)

        out["F"] = f1 + f2 + f3

class lower_level_test5(Problem):
    def __init__(self):
        self.n_var = 3
        self.n_constr = 0
        self.n_obj = 1

        xl1_u = [10] * 2
        xl1_l = [-5] * 2

        xl2_u = [10] * 1
        xl2_l = [-5] * 1

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)



    def _evaluate(self, x, out, *args, **kwargs):
        n = x.shape[0]
        xl1 = np.atleast_2d(x[:, 0:2]).reshape(n, -1)
        xl2 = np.atleast_2d(x[:, 2]).reshape(n, -1)

        f2 = np.zeros(x.shape[0])
        for i in range(1):
            f2 += (xl1[:, i + 1] - xl1[:, i] ** 2) ** 2 + (xl1[:, i] - 1) ** 2
        f3 = np.sum((xl2 ** 2) ** 2, axis=1)

        out["F"] = f2 + f3