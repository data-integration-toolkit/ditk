#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH -N 5
#SBATCH --reservation=tensorflow

export PYTHONUNBUFFERED=1

srun -n $SLURM_NNODES curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
srun -n $SLURM_NNODES python3 get-pip.py --user
srun -n $SLURM_NNODES pip3 install --user --upgrade git+https://github.com/dbaumgarten/FToDTF.git
srun -n $SLURM_NNODES mkdir -p log

NUMPS=1
PORT=7777
DOMAIN=sc.uni-leipzig.de

read -r -a NODES <<< `scontrol show hostnames $SLURM_JOB_NODELIST`
WORKERS=(${NODES[@]:0:SLURM_NNODES-NUMPS})
SLAVES=(${WORKERS[@]:1})
PS=(${NODES[@]:SLURM_NNODES-NUMPS})

createlist()
{
str=""
for i in $@; do
str=$str,$i.$DOMAIN:$PORT
done
str=${str:1}
echo $str
}

WORKERLIST=`createlist ${WORKERS[@]}`
PSLIST=`createlist ${PS[@]}`


echo Starting parameter-server: ${PS[@]}
PSINDEX=0
for i in ${PS[@]}; do
srun -N 1 -n 1 --nodelist=${PS[PSINDEX]} --job-name ps-$PSINDEX fasttext train --job ps --workers="$WORKERLIST" --ps="$PSLIST" --index=$PSINDEX &
let PSINDEX=${PSINDEX}+1
done

echo Starting worker: ${SLAVES[@]}
WORKERINDEX=1
for i in ${SLAVES[@]}; do
srun -N 1 -n 1 --nodelist=${WORKERS[WORKERINDEX]} --job-name worker-$WORKERINDEX fasttext train --job worker --workers="$WORKERLIST" --ps="$PSLIST" --index=$WORKERINDEX --batches_file batches_$WORKERINDEX.tfrecord &
let WORKERINDEX=${WORKERINDEX}+1
done

echo STARTING CHIEF-WORKER ON: ${WORKERS[0]}
srun -N 1 -n 1 --nodelist=${WORKERS[0]} --job-name master fasttext train --job worker --workers="$WORKERLIST" --ps="$PSLIST" --index=0 --batches_file batches_0.tfrecord