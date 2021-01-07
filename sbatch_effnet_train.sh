#!/bin/bash
#SBATCH -p t4v2
#SBATCH --exclude=gpu102
#SBATCH --exclude=gpu115
#SBATCH --gres=gpu:4                        # request GPU(s)
#SBATCH --qos=normal
#SBATCH -c 24                                # number of CPU cores
#SBATCH --mem=128G                           # memory per node
#SBATCH --time=500:00:00                     # max walltime, hh:mm:ss
#SBATCH --array=0%1                    # array value
#SBATCH --output=logs_new/ef_ho/%a-%N-%j    # %N for node name, %j for jobID
#SBATCH --job-name=ef_ho

source ~/.bashrc
source activate ~/venvs/efficientnet_train

SAVE_PATH="$1"
ACTFUN="$2"
SEED="$SLURM_ARRAY_TASK_ID"

touch /checkpoint/robearle/${SLURM_JOB_ID}
CHECK_PATH=/checkpoint/robearle/${SLURM_JOB_ID}

# Debugging outputs
pwd
which conda
python --version
pip freeze

echo ""
python -c "import torch; print('torch version = {}'.format(torch.__version__))"
python -c "import torch.cuda; print('cuda = {}'.format(torch.cuda.is_available()))"
python -c "import scipy; print('scipy version = {}'.format(scipy.__version__))"
python -c "import sklearn; print('sklearn version = {}'.format(sklearn.__version__))"
python -c "import matplotlib; print('matplotlib version = {}'.format(matplotlib.__version__))"
python -c "import tensorflow; print('tensorflow version = {}'.format(tensorflow.__version__))"
echo ""

echo "SAVE_PATH=$SAVE_PATH"
echo "SEED=$SEED"

./distributed_train.sh 4 /scratch/ssd001/datasets/imagenet/ --model efficientnet_b0 -b 384 --actfun $ACTFUN --output $SAVE_PATH --check-path $CHECK_PATH --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --aa original --remode pixel --reprob 0.2 --lr .096

