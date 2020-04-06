import autograd.numpy as anp
from surrogate_problems.sur_problem_base import Problem
from sklearn.utils.validation import check_array
import numpy as np
import pygmo as pg


from scipy.stats import norm
from sklearn.utils.validation import check_array
from EI_krg import expected_improvement




class EIM(Problem):

    def __init__(self, n_var, n_obj, n_constr, upper_bound, lower_bound):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(lower_bound)
        self.xu = anp.array(upper_bound)
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):

        # input should be in the right range of defined problem
        train_y = kwargs['train_y']
        norm_train_y = kwargs['norm_train_y']
        feasible = kwargs['feasible']
        nadir = kwargs['nadir']
        ideal = kwargs['ideal']
        ei_method = kwargs['ei_method']
        krg = kwargs['krg']
        krg_g = kwargs['krg_g']

        eim = expected_improvement(x,
                                   train_y,
                                   norm_train_y,
                                   feasible,
                                   nadir,
                                   ideal,
                                   ei_method,
                                   krg,
                                   krg_g,
                                   )

        out["F"] = -eim






