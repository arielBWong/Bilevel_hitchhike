import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg


class SMD1_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [10] * r
        xu2_lbound = [-5] * r

        self.xl = anp.array(xu1_ubound + xu2_ubound)
        self.xu = anp.array(xu1_lbound + xu2_lbound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        F2 = np.sum(xl1 ** 2, axis=1)
        F3 = np.sum(xu2 ** 2, axis=1) + np.sum((xu2 - np.tan(xl2))**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD1_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [np.pi/2 - 1e-8] * r
        xl2_l = [-np.pi/2 + 1e-8] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)
        f2 = np.sum(xl1 ** 2, axis=1)
        f3 = np.sum((xu2 - np.tan(xl2))**2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD2_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [1] * r
        xu2_lbound = [-5] * r

        self.xl = anp.array(xu1_lbound + xu2_lbound)
        self.xu = anp.array(xu1_ubound + xu2_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        F2 = -np.sum(xl1 ** 2, axis=1)
        F3 = np.sum(xu2 ** 2, axis=1) - np.sum((xu2 - np.log10(xl2))**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD2_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [np.e] * r
        xl2_l = [0 + 1e-8] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)
        f2 = np.sum(xl1 ** 2, axis=1)
        f3 = np.sum((xu2 - np.log10(xl2))**2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD3_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [10] * r
        xu2_lbound = [-5] * r

        self.xu = anp.array(xu1_ubound + xu2_ubound)
        self.xl = anp.array(xu1_lbound + xu2_lbound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        F2 = np.sum(xl1 ** 2, axis=1)
        F3 = np.sum((xu2-1) ** 2, axis=1) + np.sum((xu2**2 - np.tan(xl2))**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD3_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [np.pi/2-1e-8] * r
        xl2_l = [-np.pi/2+2e-8] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)
        f2 = self.q + np.sum(xl1 ** 2 - np.cos(2 * np.pi * xl1), axis=1)
        f3 = np.sum((xu2 ** 2 - np.tan(xl2))**2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD4_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [1] * r
        xu2_lbound = [-1] * r

        self.xu = anp.array(xu1_ubound + xu2_ubound)
        self.xl = anp.array(xu1_lbound + xu2_lbound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        F2 = -np.sum(xl1 ** 2, axis=1)
        F3 = np.sum(xu2 ** 2, axis=1) - np.sum((np.abs(xu2) - np.log10(1 + xl2))**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD4_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [np.e] * r
        xl2_l = [0] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)
        f2 = self.q + np.sum(xl1 ** 2 - np.cos(2 * np.pi * xl1), axis=1)
        f3 = np.sum((np.abs(xu2) - np.log10(1 + xl2)) ** 2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD5_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [10] * r
        xu2_lbound = [-5] * r

        self.xl = anp.array(xu1_lbound + xu2_lbound)
        self.xu = anp.array(xu1_ubound + xu2_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        F2 = np.zeros(x.shape[0])
        for i in range(self.q-1):
            F2 += (xl1[:, i+1] - xl1[:, i] ** 2) ** 2 + (xl1[:, i] - 1)**2
        F2 = -F2
        F3 = np.sum(xu2 ** 2, axis=1) - np.sum((np.abs(xu2) - xl2**2)**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD5_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [10] * r
        xl2_l = [-5] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        # x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)
        f2 = np.zeros(x.shape[0])
        for i in range(self.q - 1):
            f2 += (xl1[:, i+1] - xl1[:, i]**2)**2 + (xl1[:, i] - 1)**2
        f3 = np.sum((np.abs(xu2) - xl2**2)**2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD6_F(Problem):

    def __init__(self, p, r, q, s):
        self.s = s
        self.n_var = p + q + s + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0


        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [10] * r
        xu2_lbound = [-5] * r

        self.xl = anp.array(xu1_lbound + xu2_lbound)
        self.xu = anp.array(xu1_ubound + xu2_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q + self.s))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q + self.s), (self.p + self.r + self.q + self.s + self.r))]

        F1 = np.sum(xu1 ** 2, axis=1)
        #F2 = -np.sum(xl1[:, 0:self.q] ** 2, axis=1) + np.sum(xl1[:, self.q: self.q + self.s]**2, axis=1)
        F2 = -np.sum(xl1**2, axis=1) + np.sum(xl1**2, axis=1)

        F3 = np.sum(xu2 ** 2, axis=1) - np.sum((xu2 - xl2)**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD6_f(Problem):

    def __init__(self, p, r, q, s):
        self.s = s
        self.n_var = p + q + s + r * 2
        self.n_levelvar = q + s + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * (q+s)
        xl1_l = [-5] * (q+s)

        xl2_u = [10] * r
        xl2_l = [-5] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        # x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q + self.s))]
        xl2 = x[:, np.arange((self.p + self.r + self.q + self.s), (self.p + self.r + self.q + self.s + self.r))]


        f1 = np.sum(xu1 ** 2, axis=1)

        index_p1 = np.arange(self.q, self.q + self.s-1, 2)
        index_p2 = np.arange(self.q+1, self.q + self.s, 2)
        f2 = np.sum(xl1[:, 0: self.q] ** 2, axis=1) + ((xl1[:, index_p2] - xl1[:, index_p1])**2).ravel()
        f3 = np.sum((xu2 - xl2)**2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD7_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [1] * r
        xu2_lbound = [-5] * r

        self.xl = anp.array(xu1_lbound + xu2_lbound)
        self.xu = anp.array(xu1_ubound + xu2_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1_part2 = 1
        for i in range(self.p):
            F1_part2 = F1_part2 * np.cos(xu1[:, i]/np.sqrt(i+1))

        F1 = 1 + 1/400 * np.sum(xu1 ** 2, axis=1) - F1_part2
        F2 = -np.sum(xl1 ** 2, axis=1)
        F3 = np.sum(xu2 ** 2, axis=1) - np.sum((xu2 - np.log10(xl2))**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD7_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [np.e] * r
        xl2_l = [0 + 1e-8] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(xu1 ** 3, axis=1)
        f2 = np.sum(xl1 ** 2, axis=1)
        f3 = np.sum((xu2 - np.log10(xl2))**2, axis=1)

        out["F"] = f1 + f2 + f3

class SMD8_F(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        xu1_ubound = [10] * p
        xu1_lbound = [-5] * p

        xu2_ubound = [10] * r
        xu2_lbound = [-5] * r

        self.xl = anp.array(xu1_lbound + xu2_lbound)
        self.xu = anp.array(xu1_ubound + xu2_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        # x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:,  np.arange(self.p, (self.p + self.r))]

        xl1 = x[:,  np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:,  np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]

        F1 = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(1/self.p * np.sum(xu1 ** 2, axis=1))) - \
            np.exp(1/self.p * np.sum(np.cos(2 * np.pi * xu1), axis=1))
        F2 = [0] * x.shape[0]
        for i in range(self.q-1):
            F2 +=  (xl1[:, i+1] - xl1[:, i]**2)**2 + (xl1[:, i] - 1)**2
        F2 = -F2
        F3 = np.sum(xu2 ** 2, axis=1) - np.sum((xu2 - xl2**3)**2, axis=1)

        out["F"] = F1 + F2 + F3

class SMD8_f(Problem):

    def __init__(self, p, r, q):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0

        xl1_u = [10] * q
        xl1_l = [-5] * q

        xl2_u = [10] * r
        xl2_l = [-5] * r

        self.xl = anp.array(xl1_l + xl2_l)
        self.xu = anp.array(xl1_u + xl2_u)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu1 = x[:, np.arange(0, self.p)]
        xu2 = x[:, np.arange(self.p, (self.p + self.r))]

        xl1 = x[:, np.arange((self.p + self.r), (self.p + self.r + self.q))]
        xl2 = x[:, np.arange((self.p + self.r + self.q), (self.p + self.r + self.q + self.r))]


        f1 = np.sum(np.abs(xu1), axis=1)
        f2 = 0
        for i in range(self.q - 1):
            f2 += (xl1[:, i + 1] - xl1[:, i] ** 2) ** 2 + (xl1[:, i] - 1) ** 2
        f3 = np.sum((xu2 - xl2**3)**2, axis=1)

        out["F"] = f1 + f2 + f3