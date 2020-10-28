import numpy as np
from dataloader import FrameLoader
import argparse
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
import tqdm
import logging
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_video_csv',
        default='data/video_paths.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--durations_csv',
        default='data/durations.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--batch_size', 
        default=1, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--gpu_number',
        default='0',
        type=str,
        help='Window size for feature extraction')
    parser.add_argument(
        '--faces_per_frame',
        default=2,
        type=int,
        help='Max number of faces to keep per frame')
    return parser.parse_args() 

def main():
    args = get_arguments()
    
    device = th.device(f'cuda:{args.gpu_number}')
    mtcnn = MTCNN(keep_all=True,device=device)

    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    dataset = FrameLoader(videos_csv=args.path_video_csv,  durations_csv=args.durations_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)#4)

    max_faces = args.faces_per_frame
    feat_dim = 512*max_faces

    num_frames = 0
    num_faces = 0
    logging.info(f"Extracting faces feature with maximum {max_faces} faces per frame")
    with th.no_grad():
        for images, out_path in tqdm.tqdm(dataloader):
            if len(images.shape) > 3:
                images = images.squeeze(0)
                n_chunk = images.shape[0]
                num_frames =+ n_chunk
                features = th.cuda.FloatTensor(n_chunk, feat_dim).fill_(0)
                n_iter = int(np.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    images_batch = images[min_ind:max_ind]
                    try:
                        faces_batch, probs = mtcnn(images_batch, return_prob=True)
                        num_faces += len(faces_batch)
                        embeddings = th.stack([F.pad(resnet(faces[0:max_faces]).view(-1), (0,1024-(512*faces[0:max_faces].shape[0]))) 
                                    if (faces is not None and prob.any()>0.9) else th.zeros(feat_dim) for faces,prob in zip(faces_batch,probs)],dim=0)
                    except:
                        print(f"Error in processing faces of video: {out_path}")
                        continue

                    features[min_ind:max_ind] = embeddings
                features = features.cpu().numpy()
                np.save(out_path[0], features)
            else:
                print(f'Video {out_path} already processed.')
        
        logging.info(f'Done, average number of faces per frame: {num_faces/num_frames}')
        

if __name__ == "__main__":
    main()