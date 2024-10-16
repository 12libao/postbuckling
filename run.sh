####################### Optimization #######################
#### buckling
# python buckling_optimization_run.py --objf compliance --confs volume --nx 120 --r 2 --N 1 --grad-check

# python buckling_optimization_run.py --objf ks-buckling --confs volume --nx 300 --r 3 --N 6 --grad-check --note -02

# python buckling_optimization_run.py --confs volume compliance --nx 60 --r 2 --N 6 --grad-check

# python buckling_optimization_run.py --objf compliance-buckling --confs volume --nx 120 --r 4 --N 6 --grad-check --w 0.5

# python buckling_optimization_run.py --objf koiter-a --confs volume --nx 120 --r 2 --N 1 --grad-check

# python buckling_optimization_run.py --objf koiter-a --confs volume ks-buckling --nx 120 --r 2 --N 6 --BLF-ks-lb 10 --grad-check

# python buckling_optimization_run.py --objf koiter-a --confs volume ks-buckling koiter-b --nx 120 --r 2 --N 6 --BLF-ks-lb 8 --b-lb 1e-5 --grad-check

# python buckling_optimization_run.py --objf koiter-b --confs volume ks-buckling --nx 120 --r 2 --N 6 --BLF-ks-lb 10 --grad-check

# python buckling_optimization_run.py --objf koiter-al0 --confs volume ks-buckling --nx 120 --r 2 --N 6 --BLF-ks-lb 10 --grad-check

# python buckling_optimization_run.py --objf koiter-lams --confs volume ks-buckling --nx 300 --r 2 --N 6 --BLF-ks-lb 10 --grad-check

python buckling_optimization_run.py --objf koiter-ks-lams --confs volume ks-buckling --nx 300 --r 3 --N 6 --xi 1e-3 --note xi-3 --grad-check --BLF-ks-lb 8

# python buckling_optimization_run.py --objf koiter-nlams --confs volume ks-buckling koiter-b --nx 120 --r 2 --N 6 --BLF-ks-lb 8 --b-lb 1e-5 --grad-check
