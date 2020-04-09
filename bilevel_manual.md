## Where to run algorithm ##
For bilevel problems, main is in file EGO_krg.py. Find 
```python
if __name__ == "__main__": 
```

Method ‘main_bi_mo’ is where bilevel optimization process is conducted.
To run this method, it needs 5 arguments: (random seed, pair of bilevel problems(list of 2), crossvalidation flag, method_flag, save_flag)

Parallel running of different seeds is possible by uncommenting following lines

```python
num_workers = 22
pool = mp.Pool(processes=num_workers)
pool.starmap(main_bi_mo, ([arg for arg in args]))
```
