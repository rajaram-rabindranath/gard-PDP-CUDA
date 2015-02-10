#!/bin/bash

jobname="GPU_BSORT"

rm -rf ALL_SLURMS
mkdir ALL_SLURMS ## store all the SLURMs created to this location

## FIXME here is where one can set the problem size --- like so probSize=(24)
probSize=(10000) # 10000000 50000000) # 100000000 250000000)



func()
{
directive="#!/bin/bash\n\n#SBATCH --partition=$1\n#SBATCH --gres=gpu:1\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --job-name=$jobname\n#SBATCH --time=04:20:00\n#SBATCH --mail-user=rajaramr@buffalo.edu\n#SBATCH --output=result/Result_GPU_%j.out\n#SBATCH --error=result/Result_GPU_error_%j.out\n
echo \"SLURM Enviroment Variables:\"\n
echo \"Job ID = \"\$SLURM_JOB_ID\n
echo \"Job Name = \"\$SLURM_JOB_NAME\n
echo \"Job Node List = \"\$SLURM_JOB_NODELIST\n
echo \"Number of Nodes = \"\$SLURM_NNODES\n
echo \"Tasks per Nodes = \"\$SLURM_NTASKS_PER_NODE\n
echo \"CPUs per task = \"\$SLURM_CPUS_PER_TASK\n
echo \"/scratch/jobid = \"\$SLURMTMPDIR\n
echo \"submit Host = \"\$SLURM_SUBMIT_HOST\n
echo \"Subimt Directory = \"\$SLURM_SUBMIT_DIR\n
echo \n"
echo -e $directive
}

sbatch_GPU="ulimit -s unlimited\nmodule load cuda\nmodule list\nwhich nvcc\n#\n\n./cudaBSORT \$1\n\n#\necho \"All Dones!\"\n"

func "gpu" > SLURM_GPU
echo -e $sbatch_GPU >> SLURM_GPU

for size in "${probSize[@]}"
do
   :
	sbatch SLURM_GPU $size

done

mv SLURM_GPU ALL_SLURMS/
