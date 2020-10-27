import h5py as h5
import numpy as np
import os.path as osp
import glob
import tqdm
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_path',
        default='/home/pardogl/datasets/movies/',
        type=str)
    return parser.parse_args()


def npy2h5(out_path):
    features_path = (f'{out_path}/youtube/*')
    videos = glob.glob(f'{features_path}/*_faces.npy')
    print(f'{len(videos)} features found')
    features = []
    names = []
    for video in tqdm.tqdm(videos):
        feature = np.load(open(video,'rb'))
        features.append(feature)
        name = osp.basename(video).replace('_faces.npy','')
        names.append(name)

    print('Saving hdf5 file')
    with h5.File(f'{out_path}/InceptionV1_Facenet_features.h5','w') as f:
        for name, feature in tqdm.tqdm(zip(names, features), total=len(names)):
            f.create_dataset(name, data=feature, chunks=True)

if __name__ == "__main__":
    args = get_arguments()
    # out_path = '/home/pardogl/scratch/data/movies'
    npy2h5(args.out_path)
