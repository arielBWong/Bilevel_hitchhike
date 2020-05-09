import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg


class BLTP1_F(Problem):

    def __init__(self, p=1, r=1, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 1
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 0

        if p != 1 or q != 1 or r != 1:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [50] * p
        xu1_lbound = [0] * p

        xu2_ubound = [50] * r
        xu2_lbound = [0] * r

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

        F = 2 * xu1 + 2 * xu2 - 3 * xl1 - 3 * xl2 - 60
        G = xu1 + xu2 + xl1 - 2 * xl2 - 40

        out["F"] = F
        out["G"] = G

class BLTP1_f(Problem):

    def __init__(self, p=1, r=1, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 2
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 200
        if p != 1 or q != 1 or r != 1:
            raise(
                "This problem only allow two variables"
            )

        xl1_u = [20] * q
        xl1_l = [-10] * q

        xl2_u = [20] * r
        xl2_l = [-10] * r

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


        f = (xl1 - xu1 + 20) ** 2 + (xl2 - xu2 + 20) ** 2
        g = []
        g1 = 2 * xl1 - xu1 + 10
        g = np.append(g, g1)
        g2 = 2 * xl2 - xu2 + 10
        g = np.append(g, g2)
        g = np.atleast_2d(g).reshape(-1, 2, order='F')

        out["F"] = f
        out["G"] = g

class BLTP2_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 17

        if r != 0:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [2] * p
        xu1_lbound = [0] * p

        xu2_ubound = [2] * r
        xu2_lbound = [0] * r

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

        F = (xu1 - 5) ** 2 + (2 * xl1 + 1) **2
        out["F"] = F

class BLTP2_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 3
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 1
        if r != 0:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [1] * q
        xl1_l = [0] * q

        xl2_u = [1] * r
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

        f = (xl1 - 1) ** 2 - 1.5 * xu1 * xl1

        g = []
        g1 = -(3 * xu1 - xl1 - 3)
        g = np.append(g, g1)
        g2 = -(-xu1 + 0.5 * xl1 + 4)
        g = np.append(g, g2)
        g3 = -(-xu1 - xl1 + 7)
        g = np.append(g, g3)
        g = np.atleast_2d(g).reshape(-1, 3, order='F')

        out["F"] = f
        out["G"] = g

class BLTP3_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 1

        if r != 0:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [1] * p
        xu1_lbound = [0] * p

        xu2_ubound = [1] * r
        xu2_lbound = [0] * r

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

        F = (xu1 - 1) ** 2 + (xl1 - 1) ** 2
        out["F"] = F

class BLTP3_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0
        if r != 0:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [1] * q
        xl1_l = [0] * q

        xl2_u = [1] * r
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

        f = 0.5 * xl1 ** 2 + 500 * xl1 - 50 * xu1 * xl1
        out["F"] = f

class BLTP4_F(Problem):

    def __init__(self, p=1, r=0, q=2):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 2
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -1000

        if r != 0 or p != 1 or q != 2:
            raise(
                "This problem only allow three variables"
            )


        xu1_ubound = [1] * p
        xu1_lbound = [0] * p

        xu2_ubound = [1] * r
        xu2_lbound = [0] * r

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

        F = -(100 * xu1 + 1000 * np.atleast_2d(xl1[:, 0]).reshape(-1, 1))
        out["F"] = F

        g = []
        g1 = xu1.ravel() + xl1[:, 0] - xl1[:, 1] - 1
        g = np.append(g, g1)
        g2 = xl1[:, 0] + xl1[:, 1] - 1
        g = np.append(g, g2)
        g = np.atleast_2d(g).reshape(-1, 2, order='F')

        out["G"] = g

class BLTP4_f(Problem):

    def __init__(self, p=1, r=0, q=2):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 2
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = -1
        if r != 0 or p != 1 or q != 2:
            raise(
                "This problem only allow three variable each level"
            )

        xl1_u = [1] * q
        xl1_l = [0] * q

        xl2_u = [1] * r
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

        f = -np.sum(xl1, axis=1)

        g = []
        g1 = xu1.ravel() + xl1[:, 0] - xl1[:, 1] - 1
        g = np.append(g, g1)
        g2 = xl1[:, 0] + xl1[:, 1] - 1
        g = np.append(g, g2)
        g = np.atleast_2d(g).reshape(-1, 2, order='F')

        out["F"] = f
        out["G"] = g

class BLTP5_F(Problem):

    def __init__(self, p=1, r=0, q=2):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -1.4074

        if r != 0 or q != 2 or p != 1:
            raise(
                "This problem only allow two variables"
            )

        xu1_ubound = [4] * p
        xu1_lbound = [0] * p

        xu2_ubound = [4] * r
        xu2_lbound = [0] * r

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

        F = (xu1.ravel() - 1) ** 2 + 2 * xl1[:, 0] ** 2 - 2 * xu1.ravel()
        out["F"] = F

class BLTP5_f(Problem):

    def __init__(self, p=1, r=0, q=2):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 4
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 7.6172
        if r != 0 or q != 2 or p != 1:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [2] * q
        xl1_l = [0] * q

        xl2_u = [2] * r
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

        f = (2 * xl1[:, 0] - 4) ** 2 + (2 * xl1[:, 1] - 1) ** 2 + xu1.ravel() * xl1[:, 0]

        g = []
        g1 = 4 * xu1.ravel() + 5 * xl1[:, 0] + 4 * xl1[:, 1] - 12
        g = np.append(g, g1)
        g2 = -4 * xu1.ravel() - 5 * xl1[:, 0] + 4 * xl1[:, 1] + 4
        g = np.append(g, g2)
        g3 = 4 * xu1.ravel() - 4 * xl1[:, 0] + 5 * xl1[:, 1] - 4
        g = np.append(g, g3)
        g4 = -4 * xu1.ravel() + 4 * xl1[:, 0] + 5 * xl1[:, 1] - 4
        g = np.append(g, g4)

        g = np.atleast_2d(g).reshape(-1, 4, order='F')

        out["F"] = f
        out["G"] = g

class BLTP6_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 1
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 100

        if r != 0:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [15] * p
        xu1_lbound = [0] * p

        xu2_ubound = [15] * r
        xu2_lbound = [0] * r

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

        F = xu1 ** 2 + (xl1 - 10) ** 2
        G = -xu1 + xl1

        out["F"] = F
        out["G"] = G

class BLTP6_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 1
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0
        if r != 0:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [20] * q
        xl1_l = [0] * q

        xl2_u = [20] * r
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

        f = (xu1 + 2 * xl1 - 30)**2
        g = xu1 + xl1 - 20

        out["F"] = f
        out["G"] = g


class BLTP7_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -49

        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [16] * p
        xu1_lbound = [0] * p

        xu2_ubound = [16] * r
        xu2_lbound = [0] * r

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

        F = -(xu1 + 3 * xl1)
        out["F"] = F


class BLTP7_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 5
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 17
        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [12] * q
        xl1_l = [0] * q

        xl2_u = [12] * r
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

        f = -(xu1 - 3 * xl1)
        g = []
        g1 = -xu1 - 2 * xl1 + 10
        g = np.append(g, g1)
        g2 = xu1 - 2 * xl1 - 6
        g = np.append(g, g2)
        g3 = 2 * xu1 - xl1 - 21
        g = np.append(g, g3)
        g4 = xu1 + 2 * xl1 - 38
        g = np.append(g, g4)
        g5 = -xu1 + 2*xl1 - 18
        g = np.append(g, g5)

        g = np.atleast_2d(g).reshape(-1, 5, order='F')

        out["F"] = f
        out["G"] = g


class BLTP8_F(Problem):

    def __init__(self, p=1, r=1, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -1

        if r != 1 or p != 1 or q != 1:
            raise(
                "This problem only allow two variables each level"
            )


        xu1_ubound = [1.5] * p
        xu1_lbound = [0] * p

        xu2_ubound = [1.5] * r
        xu2_lbound = [0] * r

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

        F = xu1 ** 2 - 2 * xu1 + xu2 ** 2 - 2 * xu2 + xl1 ** 2 + xl2 ** 2
        out["F"] = F


class BLTP8_f(Problem):

    def __init__(self, p=1, r=1, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0
        if r != 1 or p != 1 or q != 1:
            raise(
                "This problem only allow two variable each level"
            )

        xl1_u = [1.5] * q
        xl1_l = [0.5] * q

        xl2_u = [1.5] * r
        xl2_l = [0.5] * r

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

        f = (xl1 - xu1) ** 2 + (xl2 - xu2) ** 2

        out["F"] = f




class BLTP9_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -85.0909

        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [18] * p
        xu1_lbound = [0] * p

        xu2_ubound = [18] * r
        xu2_lbound = [0] * r

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

        F = -(-2 * xu1 + 11 * xl1)
        out["F"] = F


class BLTP9_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 6
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 50.1818
        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [11] * q
        xl1_l = [0] * q

        xl2_u = [11] * r
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

        f = -(-xu1 - 3 * xl1)
        g = []
        g1 = xu1 - 2 * xl1 - 4
        g = np.append(g, g1)
        g2 = 2 * xu1 - xl1 - 24
        g = np.append(g, g2)
        g3 = 3 * xu1 + 4 * xl1 - 96
        g = np.append(g, g3)
        g4 = xu1 + 7 * xl1 - 126
        g = np.append(g, g4)
        g5 = -4 * xu1 + 5 * xl1 - 65
        g = np.append(g, g5)
        g6 = -(xu1 + 4 * xl1 - 8)
        g = np.append(g, g6)

        g = np.atleast_2d(g).reshape(-1, 6, order='F')

        out["F"] = f
        out["G"] = g



class BLTP10_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 5

        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [8] * p
        xu1_lbound = [0] * p

        xu2_ubound = [8] * r
        xu2_lbound = [0] * r

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

        F = (xu1 - 3)**2 + (xl1 - 2)**2
        out["F"] = F


class BLTP10_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 3
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 4
        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [40] * q
        xl1_l = [0] * q

        xl2_u = [40] * r
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

        f = (xl1 - 5)**2
        g = []
        g1 = -(2 * xu1 - xl1 + 1)
        g = np.append(g, g1)
        g2 = -(-xu1 + 2 * xl1 - 2)
        g = np.append(g, g2)
        g3 = -(-xu1 - 2 * xl1 + 14)
        g = np.append(g, g3)

        g = np.atleast_2d(g).reshape(-1, 3, order='F')

        out["F"] = f
        out["G"] = g




class BLTP11_F(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 3
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = 9

        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow two variables"
            )


        xu1_ubound = [8] * p
        xu1_lbound = [0] * p

        xu2_ubound = [8] * r
        xu2_lbound = [0] * r

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

        F = (xu1 - 3)**2 + (xl1 - 2)**2
        out["F"] = F

        g = []
        g1 = -(2 * xu1 - xl1 + 1)
        g = np.append(g, g1)
        g2 = -(-xu1 + 2 * xl1 - 2)
        g = np.append(g, g2)
        g3 = -(-xu1 - 2 * xl1 + 14)
        g = np.append(g, g3)

        g = np.atleast_2d(g).reshape(-1, 3, order='F')
        out["G"] = g



class BLTP11_f(Problem):

    def __init__(self, p=1, r=0, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 0
        if r != 0 or p != 1 or q != 1:
            raise(
                "This problem only allow one variable each level"
            )

        xl1_u = [50] * q
        xl1_l = [0] * q

        xl2_u = [50] * r
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

        f = (xl1 - 5)**2
        out["F"] = f


class BLTP12_F(Problem):

    def __init__(self, p=1, r=1, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -3.25

        if r != 1 or p != 1 or q != 1:
            raise(
                "This problem only allow default variables"
            )


        xu1_ubound = [2] * p
        xu1_lbound = [0] * p

        xu2_ubound = [2] * r
        xu2_lbound = [0] * r

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

        F = -(2 * xu1 - xu2 - 0.5 * xl1)
        out["F"] = F




class BLTP12_f(Problem):

    def __init__(self, p=1, r=1, q=1):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 3
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = -4
        if r != 1 or p != 1 or q != 1:
            raise(
                "This problem only default variable each level"
            )

        xl1_u = [2] * q
        xl1_l = [0] * q

        xl2_u = [2] * r
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

        f = -(-xu1- xu2 + 4 * xl1 - xl2)
        out["F"] = f

        g = []
        g1 = 2 * xu1 - xl1 + xl2 - 2.5
        g1 = -g1
        g = np.append(g, g1)
        g2 = -xu1 + 3 * xu2 - xl2 + 2
        g2 = -g2
        g = np.append(g, g2)
        g3 = -xu1 - xu2 + 2
        g3 = -g3
        g = np.append(g, g3)

        g = np.atleast_2d(g).reshape(-1, 3, order='F')
        out["G"] = g




class BLTP13_F(Problem):

    def __init__(self, p=2, r=0, q=3):
        self.n_var = p + q + r * 2
        self.n_levelvar = p + r
        self.n_constr = 0
        self.n_obj = 1
        self.p = p
        self.r = r
        self.q = q
        self.opt = -29.2

        if r != 0 or p != 2 or q != 3:
            raise(
                "This problem only allow default variables"
            )


        xu1_ubound = [1] * p
        xu1_lbound = [0] * p

        xu2_ubound = [1] * r
        xu2_lbound = [0] * r

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

        F = 8 * xu1[:, 0] + 4 * xu1[:, 1] - 4 * xl1[:, 0] + 40 * xl1[:, 1] + 4 * xl1[:, 2]
        out["F"] = -F




class BLTP13_f(Problem):

    def __init__(self, p=2, r=0, q=3):
        self.n_var = p + q + r * 2
        self.n_levelvar = q + r
        self.n_constr = 3
        self.n_obj = 1
        self.p = p
        self.q = q
        self.r = r
        self.opt = 3.2
        if r != 0 or p != 2 or q != 3:
            raise(
                "This problem only allow default variable each level"
            )

        xl1_u = [1] * q
        xl1_l = [0] * q

        xl2_u = [1] * r
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

        f = -xu1[:, 0] - 2 * xu1[:, 1] - xl1[:, 0] - xl1[:, 1] - 2 * xl1[:, 2]
        out["F"] = -f

        g = []
        g1 = -xl1[:, 0] + xl1[:, 1] + xl1[:, 2] - 1
        g = np.append(g, g1)
        g2 = 2 * xu1[:, 0] - xl1[:, 0] + 2 * xl1[:, 1] - 0.5 * xl1[:, 2] - 1
        g = np.append(g, g2)
        g3 = 2 * xu1[:, 1] + 2 * xl1[:, 0] - xl1[:, 1] - 0.5*xl1[:, 2] - 1
        g = np.append(g, g3)

        g = np.atleast_2d(g).reshape(-1, 3, order='F')
        out["G"] = g