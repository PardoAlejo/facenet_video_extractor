#!/bin/bash
#SBATCH --job-name FaceV
#SBATCH --array=1-25
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 96GB
##SBATCH --mail-type=ALL
##SBATCH --mail-user=alejandro.pardo@kaust.edu.sa
##SBATCH --constraint=[v100]
##SBATCH -A conf-gpu-2020.11.23

echo `hostname`
# conda activate refineloc
# module load anaconda3
source activate facenet

# DIR=$HOME/facenet_feature_extractor
# cd $DIR
echo `pwd`

python extract.py --batch_size=128 --gpu_number=0 --path_video_csv data/video_paths.csv
# python npy2hdf5.py  --out_path '/home/pardogl/scratch/data/movies/'