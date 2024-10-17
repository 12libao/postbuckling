#!/bin/bash
#SBATCH --job-name=srun
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=24:00:00
#SBATCH -o out/Report-%j.out
#SBATCH -e out/Report-%j.err

# Set environment variables to address the warning
export MV2_USE_ALIGNED_ALLOC=1
export MV2_USE_THREAD_WARNING=0


#### column
# srun python3 -u buckling_optimization_run.py --domain column --objf compliance-buckling --confs volume --vol-frac-ub 0.3 --nx 100 --r 4 --N 6 --grad-check

# srun python3 -u buckling_optimization_run.py --domain column --objf ks-buckling --confs volume compliance --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --grad-check

# srun python3 -u buckling_optimization_run.py --domain column --objf koiter-ks-lams-b --confs volume compliance --vol-frac-ub 0.3 --nx 100 --r 3 --N 6 --xi 1e-3 --grad-check

# srun python3 -u buckling_optimization_run.py --domain column --objf koiter-ks-lams-bc --confs volume --vol-frac-ub 0.3 --nx 100 --r 3 --N 6 --xi 1e-1 --grad-check


#### rooda
# srun python3 -u buckling_optimization_run.py --objf ks-buckling --confs volume --nx 300 --r 3 --N 6 --grad-check --case-num $SLURM_JOB_ID

# srun python3 -u buckling_optimization_run.py --objf koiter-ks-lams --confs volume ks-buckling --nx 300 --r 3 --N 6 --xi 5e-5 --grad-check --BLF-ks-lb 9 --case-num $SLURM_JOB_ID

# srun python3 -u buckling_optimization_run.py --objf koiter-ks-lams --confs volume --nx 600 --r 3 --N 6 --xi 1e-2 --grad-check --case-num $SLURM_JOB_ID