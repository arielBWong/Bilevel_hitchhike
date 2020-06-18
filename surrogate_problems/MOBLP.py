import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np


class MOBP1_F(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = p
        self.n_constr = 2
        self.n_obj = 2
        self.p = p  # upper
        self.q = q  # lower
        self.opt = 0 #？？

        xu_ubound = [10] * p # ？？
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
            # 2d extracts one column generates 1d array
            F1 = (xu[:, 0] + 2 * xl[:, 1] + 3) * (3 * xl[:, 0] + 2)
            F2 = (2 * xu[:, 0] + xl[:, 0] + 2) * (xl[:, 1] + 1)

            out["F"] = anp.column_stack([-F1, -F2])

            G1 = 3 * xu[:, 0] + xl[:, 0] + 2 * xl[:, 1] - 5
            G2 = xl[:, 0] + xl[:, 1] - 3
            out["G"] = anp.column_stack([G1, G2])


class MOBP1_f(Problem):
    def __init__(self, p=1, q=2):
        self.n_var = p + q
        self.n_levelvar = q
        self.n_constr = 2
        self.n_obj = 1
        self.p = p  # upper
        self.q = q  # lower
        self.opt = 0 #？？

        xl_ubound = [10] * q # ？？
        xl_lbound = [0] * q

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

        f = (xl[:, 0] + 1) * (xu[:, 0] + xl[:, 0] + xl[:, 1] + 3)
        out["F"] = -f

        g1 = xu[:, 0] + 2 * xl[:, 0] + xl[:, 1] - 2
        g2 = 3 * xl[:, 0] + 2 * xl[:, 1] - 6
        out["G"] = anp.column_stack([g1, g2])

# class MOBP2_F(Problem):
