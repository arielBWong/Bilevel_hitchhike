# How to run bilevel code #
The main entrance for bilevel optimization is in python file "EGO_krg. In the main method of this file, this following code is used 
to run bilevel optimization


```python
problems_json = 'p/bi_problems'
    with open(problems_json, 'r') as data_file:
         hyp = json.load(data_file)
    target_problems = hyp['BO_target_problems']
    methods_ops = hyp['methods_ops']
    alg_settings = hyp['alg_settings']

    para_run = False
    if para_run:
        seed_max = 1
        args = paral_args_bi(target_problems, seed_max, False, methods_ops, alg_settings)
        num_workers = 6
        pool = mp.Pool(processes=num_workers)
        pool.starmap(main_bi_mo, ([arg for arg in args]))
    else:
        i = 0
        main_bi_mo(0, target_problems[i:i+2], False, 'eim', alg_settings)
```
Problem introduced to code and algorithm parameter settings are stored in jason files located in folder p. Parallel running can
be enabled by set `para_run=True`. In parallel running, another parameter need to be set is `num_workers`, which decide how many 
processors to use. If only one problem is tested, then turn `para_run=False`.



