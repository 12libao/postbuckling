#!/bin/bash
#SBATCH --job-name=srun
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=32gb
#SBATCH --time=24:00:00
#SBATCH -o out/Report-%j.out
#SBATCH -e out/Report-%j.err

# Set environment variables to address the warning
export MV2_USE_ALIGNED_ALLOC=1
export MV2_USE_THREAD_WARNING=0


#### column
# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf compliance-buckling --confs volume --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --w 0.48 --note w=0.48

srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf compliance-buckling --confs volume koiter-b --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --w 0.5 --b-lb=10 --note b=10,w=0.5


# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf ks-buckling --confs volume compliance --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --c-ub 3.9 --note c-ub=3.9

# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf ks-buckling --confs volume compliance koiter-b --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --c-ub 4.0 --b-lb=1e-3 --note b-lb=3,c-ub=4.0


# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf koiter-ks-lams-b --confs volume compliance --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --xi 5e-5 --c-ub 4.0 --note c=4.0

# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf koiter-ks-lams-b --confs volume compliance koiter-b --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --xi 5e-5 --b-lb=0 --c-ub 4.5 --note b-lb=0,c=4.5


# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf koiter-ks-lams-bc --confs volume --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --xi 5e-5 --w 0.5 --note w=0.5

# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf koiter-ks-lams-bc --confs volume koiter-b --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --xi 1e-5 --w 0.1 --b-lb=-1e-7 --note b=-7,w=0.1


# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --domain column --objf koiter-ks-lamc-b --confs volume compliance --vol-frac-ub 0.3 --nx 200 --r 4 --N 6 --c-ub 4.0 --note c=4.0



#### rooda
# srun python3 -u buckling_optimization_run.py --objf ks-buckling --confs volume --nx 300 --r 3 --N 6 --case-num $SLURM_JOB_ID

# srun python3 -u buckling_optimization_run.py --objf koiter-ks-lams --confs volume ks-buckling --nx 300 --r 3 --N 6 --xi 5e-5 --BLF-ks-lb 9 --case-num $SLURM_JOB_ID

# srun python3 -u buckling_optimization_run.py --objf koiter-ks-lams --confs volume --nx 600 --r 3 --N 6 --xi 1e-2 --case-num $SLURM_JOB_ID



#### plot arc-length
# srun python3 -u buckling.py column xib 0 s 3 file it_600

# srun python3 -u buckling.py rooda xib 0 s 3