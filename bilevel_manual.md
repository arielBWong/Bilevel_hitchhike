## Where to run algorithm ##
For bilevel problems, main is in file EGO_krg.py. Find 
```python
if __name__ == "__main__": 
```

Method 
```python
main_bi_mo
```
is where bilevel optimization process is conducted.
To run this method, it needs 5 arguments: (random seed, pair of bilevel problems(list of 2), crossvalidation flag, method_flag, save_flag)
Since the bilevel problems SMDs are stored in the list varible 
``` python 
BO_target_problems
```
An example of one run on one problem is as follows:

```python
i = 0
main_bi_mo(0, BO_target_problems[i:i+2], False, 'eim', 'eim')
```



Parallel running of different seeds is possible by uncommenting following lines.
parameters are already structured in main section

```python
num_workers = 22
pool = mp.Pool(processes=num_workers)
pool.starmap(main_bi_mo, ([arg for arg in args]))
```

## bug fix log ##
Compatible problem with BLTP5. BLTP5 is a problem only having lower level constraints.
Previous version had two logic bugs. 
1. One is at the matching xl search, wrong logic is thinking local search always return feasible, if surrogate gives it a feasible start. In fact, it can still return infeasible even with a feasible start.
2.  At the upper level adding new train x step, I used the following condition. It ignored that upper level is unconstraint problem while lower level is constraint. So it will take both feasible and infeasible solutions to build surrogate
```python
if feasible_flag is False and problem.n_const>0:
```

