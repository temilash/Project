#R single core submission script

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=01:15:00


#$ -l h_vmem=32G

#$ -l coproc_v100=2

#Get email at start and end of the job
#$ -m be

#$ -N convtasnet+train

#Now run the job
source ~/miniconda3/etc/profile.d/run.sh

module load cuda

nvidia-smi -L

conda activate clarity

python train.py