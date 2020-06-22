import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np

# checking list
# (1) min max
# (2) p = 1 xu is already 1d
# (3) x range
# (4) check number of contraints
# Problem 9 sum generate 1d?


class MOBP1_F(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 2
        self.n_obj = 2
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN  # multi objective needs no single optimum

        xu_ubound = [10] * p  # decision on using [0, 10] is based feasible region on xu and xl [0, 10] covers feasible region
        xu_lbound = [0] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)

        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        # xu only has one variable
        # 2d extracts one column generates 2d array, checked
        F1 = (xu[:, 0] + 2 * xl[:, 1] + 3) * (3 * xl[:, 0] + 2)
        F2 = (2 * xu[:, 0] + xl[:, 0] + 2) * (xl[:, 1] + 1)

        out["F"] = anp.column_stack([-F1, -F2])  # max/min checked

        G1 = 3 * xu[:, 0] + xl[:, 0] + 2 * xl[:, 1] - 5
        G2 = xl[:, 0] + xl[:, 1] - 3
        out["G"] = anp.column_stack([G1, G2])  # max/min checked


class MOBP1_f(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 2
        self.n_obj = 1
        self.p = p  # upper
        self.q = q  # lower
        self.opt = 0  #？？

        xl_ubound = [10] * q  # decision on using [0, 10] is based feasible region on xu and xl [0, 10] covers feasible region
        xl_lbound = [0] * q
        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]     # 1 variable become 1d
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f = (xl[:, 0] + 1) * (xu[:, 0] + xl[:, 0] + xl[:, 1] + 3)
        out["F"] = -f   # min/max checked

        g1 = xu[:, 0] + 2 * xl[:, 0] + xl[:, 1] - 2
        g2 = 3 * xl[:, 0] + 2 * xl[:, 1] - 6
        out["G"] = anp.column_stack([g1, g2])  # max/min checked



class MOBP2_F(Problem):
    def __init__(self, p=1, q=1):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # checked contraints
        self.n_obj = 2
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [40] * p  # determined by violating contraints f
        xu_lbound = [0] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]
        F1 = -2 * xu
        F2 = -xu + 5 * xl
        out["F"] = anp.column_stack([-F1, -F2])  # checked max/min




class MOBP2_f(Problem):
    def __init__(self, p=1, q=1):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 6  # checked constraint
        self.n_obj = 1     # checked obj
        self.p = p   # upper
        self.q = q   # lower
        self.opt = np.NaN

        xl_ubound = [40] * q  # determined by violating contraints f to a large number
        xl_lbound = [0] * q

        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f1 = -xl

        out["F"] = -f1  # min/max checked

        g1 = xu - 2 * xl - 4
        g2 = 2 * xu - xl - 24
        g3 = 3 * xu + 4 * xl - 96
        g4 = xu + 7 * xl - 126
        g5 = -4 * xu + 5 * xl - 65
        g6 = -(xu + 4 * xl - 8)   # min/max checked
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6])


class MOBP3_F(Problem):
    def __init__(self, p=2, q=2):

        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # checked constraints
        self.n_obj = 2      # checked objectives
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [40] * p   # determined by violating constraints
        xu_lbound = [0] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        F1 = 2 * xu[:, 0] - 4 * xu[:, 1] + xl[:, 0] - xl[:, 1]
        F2 = -xu[:, 0] + 2 * xu[:, 1] - xl[:, 0] + 5 * xl[:, 1]
        out["F"] = anp.column_stack([-F1, -F2])   # min/max  checked



class MOBP3_f(Problem):
    def __init__(self, p=2, q=2):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 2  #  constraints checked
        self.n_obj = 1  # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN  # not sure about this?

        xl_ubound = [40] * q  # ？？
        xl_lbound = [0] * q

        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f1 = 3 * xl[:, 0] + xl[:, 1]
        out["F"] = -f1   # min/max checked

        g1 = 4 * xu[:, 0] + 3 * xu[:, 1] + 2 * xl[:, 0] + xl[:, 1] - 60
        g2 = 2 * xu[:, 0] + xu[:, 1] + 3 * xl[:, 0] + 4 * xl[:, 1] - 60

        out["G"] = anp.column_stack([g1, g2])  # min/max checked


class MOBP5_F(Problem):
    def __init__(self, p=1, q=1):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # constraint checked
        self.n_obj = 2     # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [15] * p  # as definition
        xu_lbound = [0] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        F1 = -xu - xl
        F2 = xu ** 2 + (xl - 10)**2

        out["F"] = anp.column_stack([F1, F2])  # max/min checked


class MOBP5_f(Problem):
    def __init__(self, p=1, q=1):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 1
        self.n_obj = 1
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xl_ubound = [15] * q  # as defined
        xl_lbound = [0] * q

        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f1 = xl * (xu - 30)
        out["F"] = f1      # min/max checked

        g1 = xl - xu
        out["G"] = g1       # min/max checked




class MOBP7_F(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0
        self.n_obj = 2
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [10] * p        # as defined
        xu_lbound = [0] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        F1 = xl[:, 0] + xl[:, 1] ** 2 + xu[:, 0] + (np.sin(xl[:, 0] + xu[:, 0])) ** 2
        F2 = np.cos(xl[:, 1]) * (0.1 + xu[:, 0]) * np.exp(-(xl[:, 0]/(0.1 + xl[:, 1])))

        out["F"] = anp.column_stack([F1, F2])  # max/min checked


class MOBP7_f(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 3  #constraint checked
        self.n_obj = 1  # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xl_ubound = [10] * q  # not sure about definition
        xl_lbound = [0] * q

        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f1 = ((xl[:, 0] - 2) ** 2 + (xl[:, 1] - 1) ** 2)/4
        f2 = (xl[:, 1] * xu[:, 0] + (5 - xu[:, 0]) ** 2)/16
        f3 = np.sin(xl[:, 1]/10)
        f4 = (xl[:, 0]**2 + (xl[:, 1] - 6)**4 - 2 * xl[:, 0] * xu[:, 0] - (5 - xu[:, 0])**2)/80

        f = f1 + f2 + f3 + f4

        out['F'] = f   # min/max checked

        g1 = xl[:, 0]**2 - xl[:, 1]
        g2 = 5 * xl[:, 0] ** 2 + xl[:, 1] - 10
        g3 = xl[:, 1] + xu[:, 0]/6 - 5

        out["G"] = anp.column_stack([g1, g2, g3])  # max/min checked





class MOBP8_F(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # checked
        self.n_obj = 1    # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [2] * p  # as defined
        xu_lbound = [-1] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        F1 = (xl[:, 0] - 1) ** 2 + xl[:, 1] ** 2 + xu[:, 0] ** 2
        F2 = (xl[:, 0] - 1) ** 2 + xl[:, 1] ** 2 + (xu[:, 0] - 1)**2

        out["F"] = anp.column_stack([F1, F2])   # max/min checked




class MOBP8_f(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 0  # cons checked
        self.n_obj = 1  # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xl_ubound = [2] * q  # ？？
        xl_lbound = [-1] * q

        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f = (xl[:, 0] - xu[:, 0]) ** 2 + xl[:, 1] ** 2

        out['F'] = f     # min checked



class MOBP9_F(Problem):
    def __init__(self, p=1, q=14):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # checked constraint
        self.n_obj = 2     # checked obj
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [2] * p
        xu_lbound = [-1] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]

        F1 = (xl[:, 0] - 1) ** 2 + np.sum(xl[:, 1:] ** 2, axis=1) + xu[:, 0] ** 2
        F2 = (xl[:, 0] - 1) ** 2 + np.sum(xl[:, 1:] ** 2, axis=1) + (xu[:, 0] - 1) ** 2

        out["F"] = anp.column_stack([F1, F2])  # min checked




class MOBP9_f(Problem):
    def __init__(self, p=1, q=14):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 0
        self.n_obj = 1
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xl_ubound = [2] * q  # as defined
        xl_lbound = [-1] * q

        self.xl = anp.array(xl_lbound)
        self.xu = anp.array(xl_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]  # p=1, this xu is 1d after extraction
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f = (xl[:, 0] - xu[:, 0]) ** 2 + np.sum(xl[:, 1:] ** 2, axis=1)

        out['F'] = f  # min checked



class MOBP10_F(Problem):
    def __init__(self, p=5, q=1):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # cons checked
        self.n_obj = 2      # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = 0  # ？？

        xu_ubound = [1, 5, 5, 5, 5]
        xu_lbound = [-1, -5, -5, -5, -5]

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)] # xl has one variable so this is 1d

        F1 = (1 - xu[:, 0]) * (1 + xu[:, 1]**2 + xu[:, 2] ** 2) * xl[:, 0]
        F2 = xu[:, 0] * (1 + xu[:, 1]**2 + xu[:, 2] ** 2) * xl[:, 0]

        out["F"] = anp.column_stack([F1, F2])  # min checked


class MOBP10_f(Problem):
    def __init__(self, p=5, q=1):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 1
        self.n_obj = 1
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [2]  # as defined
        xu_lbound = [1]

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]  # p=1, this xu is 1d after extraction
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f = (1 - xu[:, 0]) * (1 + xu[:, 3]**2 + xu[:, 4] ** 2) * xl[:, 0]

        out['F'] = f  # min checked

        g = (1 - xu[:, 0]) * xl[:, 0] + 1/2 * xu[:, 0] * xl[:, 0] - 1
        g = -g   # max checked

        out["G"] = np.atleast_2d(g).reshape(-1, 1)



class MOBP11_F(Problem):
    def __init__(self, p=10, q=10):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 0  # checked
        self.n_obj = 2  # checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        xu_ubound = [1] * p  # as defined
        xu_lbound = [-1] * p

        self.xl = anp.array(xu_lbound)
        self.xu = anp.array(xu_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]
        xl = x[:, np.arange(self.p, self.p + self.q)]  # xl has one variable so this is 1d

        F1 = np.sum(np.exp(-xu/(1 + np.abs(xl))), axis=1) + np.sum(np.sin(xu/(1 + np.abs(xl))), axis=1)
        F2 = np.sum(np.exp(-xl/(1 + np.abs(xu))), axis=1) + np.sum(np.sin(xl/(1 + np.abs(xu))), axis=1)

        out["F"] = anp.column_stack([F1, F2]) # min checked


class MOBP11_f(Problem):
    def __init__(self, p, q):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = q    # cons checked
        self.n_obj = 1      # obj checked
        self.p = p  # upper
        self.q = q  # lower
        self.opt = np.NaN

        x_ubound = [1] * q
        x_lbound = [-1] * q

        self.xl = anp.array(x_lbound)
        self.xu = anp.array(x_ubound)

        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        x = check_array(x)
        xu = x[:, np.arange(0, self.p)]  # p=1, this xu is 1d after extraction
        xl = x[:, np.arange(self.p, self.p + self.q)]

        f = np.sum(np.cos(np.abs(xu) * xl), axis=1) + np.sum(np.sin(xu-xl), axis=1)

        out['F'] = f

        g = []
        for i in range(self.q):
            gi = xu[:, i] + xl[:, i] - 1
            g = np.append(g, gi)



        out["G"] = np.atleast_2d(g).reshape(-1, self.q, order='F') # min checked



if __name__ == "__main__":
    problem = MOBP11_f(p=1, q=1)
    xu = np.atleast_2d([[1], [1]])
    xl = np.atleast_2d([[-1], [1]])
    x = np.hstack((xu, xl))
    F, G = problem.evaluate(x, return_values_of=['F', 'G'])
    print(F)
    print(G)







