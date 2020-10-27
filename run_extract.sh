#!/bin/bash
#SBATCH --job-name Res152
#SBATCH --array=0-9
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 96GB
#SBATCH --constraint=[v100]
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alejandro.pardo@kaust.edu.sa
##SBATCH -A conf-gpu-2020.11.23

echo `hostname`
# conda activate refineloc
# module load anaconda3
source activate facenet

DIR=$HOME/facenet_feature_extractor
cd $DIR
echo `pwd`

# python extract.py --batch_size=128 --gpu_number=0 --path_video_csv data/video_paths_dummy.csv
python npy2hdf5.py  #--out_path '/home/pardogl/scratch/data/movies/'