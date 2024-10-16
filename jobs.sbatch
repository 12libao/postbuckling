#!/bin/bash
#SBATCH --job-name=srun
#SBATCH --account=gts-gkennedy9-coda20
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=48:00:00
#SBATCH -o out/Report-%j.out
#SBATCH -e out/Report-%j.err

# Set environment variables to address the warning
export MV2_USE_ALIGNED_ALLOC=1
export MV2_USE_THREAD_WARNING=0


######################### Optimization #########################

##### buckling
# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --objf compliance --confs volume --nx 256 --r 4 --N 10

# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --objf ks-buckling --confs volume --nx 256 --r 4 --N 30

# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --objf ks-buckling --confs volume compliance aggregate --nx 256 --r 4 --N 30

# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --objf compliance-buckling --confs volume --nx 256 --r 4 --N 30 --ks0 0.055 --note ks0-0.055

# --adjoint-method ["gmres", "pcpg", "shift-invert","approx-lanczos","ad-adjoint"] 
# --solver-type [IRAM, BasicLanczos] --lanczos-guess [0, 1] --update-guess [0, 1] 
# [--c0 7.4e-6 --ks0 0.09] [--c0 1e-5 --ks0 0.06]
# srun python3 -u buckling_optimization_run.py --case-num $SLURM_JOB_ID --objf compliance-buckling --confs volume aggregate --nx 200 --r 4 --N 30 --c0 1e-5 --ks0 0.06 --h-ub 2.0 --solver-type BasicLanczos --adjoint-method ad-adjoint --lanczos-guess 1 --update-guess 0 --sigma 29 --sigma-scale 1.25 --note m2-BasicLanczos-DL