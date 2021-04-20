#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH -a 0-8%4
#SBATCH --qos normal 
#SBATCH -p p100
#SBATCH -o slurm-%A_%a_hmm_mcar.out
#SBATCH --error slurm-%A_%a_hmm_mcar.out
#SBATCH --open-mode=append
# Make this a dataset x Sampling hparam + run evaluation script for random samples.

pcts=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seed_array=(69 420 1337)
for j in {0..8}
  do
    let task_id=$j

    if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
    then
      pct=${pcts[$j]}
      source /pkgs/anaconda3/bin/activate probml
      
      for i in {0..2}
      do
        rand_seed=${seed_array[i]}
        python3 -u run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset markov_chain  --latent-ode --noise-weight 0.01 \
        --mcar --p-miss $pct --random-seed $rand_seed --preempt-path /checkpoint/${USER}/${SLURM_JOB_ID}/${i}_${j}
      done

    fi
  done


