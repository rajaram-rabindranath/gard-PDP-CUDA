#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=DEV_DETAILS
#SBATCH --time=03:40:00
#SBATCH --mail-user=chaofeng@buffalo.edu
#SBATCH --output=Result_Vector_Sum_2.out
#SBATCH --error=Result_Vector_Sum_2.out

echo "SLURM Environment Variables:"
echo "Job ID = "$SLURM_JOB_ID
echo "Job Name = "$SLURM_JOB_NAME
echo "Job Node List = "$SLURM_JOB_NODELIST
echo "Number of Nodes = "$SLURM_NNODES
echo "Tasks per node = "$SLURM_NTASKS_PER_NODE
echo "CPUs per task = "$SLURM_CPUS_PER_TASK
echo "/scratch/jobid = "$SLURMTMPDIR
echo "Submit Host = "$SLURM_SUBMIT_HOST
echo "Submit Directory = "$SLURM_SUBMIT_DIR
echo 
echo

ulimit -s unlimited
module load cuda
which nvcc
#
./cudaPrime adad

#
echo "All Done!"
