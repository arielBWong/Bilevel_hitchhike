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
