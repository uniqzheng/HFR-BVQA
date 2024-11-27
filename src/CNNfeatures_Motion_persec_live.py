"""Extracting Video Motion Features using model-based transfer learning"""

from argparse import ArgumentParser
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
import h5py
import numpy as np
import random
import time
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, videos_dir, video_names, score, video_format, width=None, height=None, framerate=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height
        self.framerate = framerate

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        #ormat = self.format[idx]
        # assert self.format == 'YUV420' or self.format == 'RGB'
        # if format == 'yuv420p':
        #     # BVI_HFR
        # dir_str = video_name.split('-')
        # num_str = dir_str[1].split('Hz')
        # dir_path = num_str[0] + 'fps'
        # if num_str[0] == '120':
        #     video_realname = dir_str[0] + '_crf0_yuv420_2K_120fps.yuv'
        # else:
        #     video_realname = dir_str[0] + '-' + num_str[0] + 'fps-360-1920x1080.yuv'
        # video_data = skvideo.io.vread(os.path.join(self.videos_dir, dir_path, video_realname),
        #                               int(self.height[idx]), int(self.width[idx]),
        #                               inputdict={'-pix_fmt': 'yuv420p'})
        #     # BVI_HFR_Done
        #
        #     # YouTube-UGC
        #     # video_realname = video_name + '_crf_10_ss_00_t_20.0.mp4'
        #     # video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_realname),
        #     #                               int(self.height[idx]), int(self.width[idx]),
        #     #                               inputdict={'-pix_fmt': 'yuv420p'})
        #
        #     # YouTube-UGC Done
        #     # LIVE_HFR
        #     dir_path = video_name.split('_')
        #     video_realname = video_name + '.webm'
        #     video_data = skvideo.io.vread(os.path.join(self.videos_dir, dir_path[0], video_realname),
        #                                   int(self.height[idx]), int(self.width[idx]),
        #                                   inputdict={'-pix_fmt': 'yuv420p'})
        #     # LIVE_HFR Done
        # elif format == 'yuvj420p':
        #     video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name), self.height[idx],
        #                                   self.width[idx], inputdict={'-pix_fmt': 'yuvj420p'})
        # elif format == 'yuv420p10le':
        #     # LIVE_HFR
        dir_path = video_name.split('_')
        video_realname = video_name + '.webm'
        # video_data = skvideo.io.vread(os.path.join(self.videos_dir, dir_path[0], video_realname))
        #     # LIVE_HFR Done
        # else:
        #     video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name))
        # video_realname = str(int(video_name)) + '.mp4'
        # video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_realname))
        # video_data_fromskv = []
        # for i in range(video_data.shape[0]):
        #     cur_frame = video_data[i,:,:,:]
        #     the_ratio = min(cur_frame.shape[0], cur_frame.shape[1]) / 512
        #     newshape0 = cur_frame.shape[0] / the_ratio
        #     newshape1 = cur_frame.shape[1] / the_ratio
        #     cur_newframe = cv2.resize(cur_frame, (int(newshape1), int(newshape0)))
        #     video_data_fromskv.append(cur_newframe)
        # video_data_fromskv = np.array(video_data_fromskv)
        # Using CV2
        videoCapture = cv2.VideoCapture(os.path.join(self.videos_dir, dir_path[0], video_realname))
        video_data_fromcv2 = []
        while True:
            success, frame = videoCapture.read()
            if success:
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                the_ratio = min(frame.shape[0], frame.shape[1])/512
                newshape0 = frame.shape[0]/the_ratio
                newshape1 = frame.shape[1]/the_ratio
                newframe = cv2.resize(frame, (int(newshape1), int(newshape0)))
                video_data_fromcv2.append(newframe)
            else:
                break
        video_data_fromcv2 = np.array(video_data_fromcv2)

        video_score = self.score[idx]
        video_framerate = self.framerate[idx]

        video_height = video_data_fromcv2.shape[1]
        video_width = video_data_fromcv2.shape[2]
        print('video_width: {} video_height: {}'.format(video_width, video_height))

        sample = {'video': video_data_fromcv2, 'score': video_score, 'framerate': video_framerate}

        return sample


class CNNModel(torch.nn.Module):
    """Modified CNN models for feature extraction"""

    def __init__(self, model='ResNet-50'):
        super(CNNModel, self).__init__()
        if model == 'MotionExtractor':
            print("use MotionExtractor")
            from MotionExtractor.get_motionextractor_model import make_motion_model
            model = make_motion_model()
            self.features = model
            self.model = 'MotionExtractor'
        else:
            print("use default ResNet-50")
            self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
            self.model = 'ResNet-50'

    def forward(self, x):
        x = self.features(x)

        if self.model == 'MotionExtractor':
            features_mean = nn.functional.adaptive_avg_pool2d(x[1], 1)
            features_std = global_std_pool3d(x[1])
            features_mean = torch.squeeze(features_mean).permute(1, 0)
            features_std = torch.squeeze(features_std).permute(1, 0)
        else:
            features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
            features_std = global_std_pool2d(x)

        return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)


def global_std_pool3d(x):
    """3D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], x.size()[2], -1, 1), dim=3, keepdim=True)


from MotionExtractor.slowfast.visualization.utils import process_cv2_inputs
from MotionExtractor.slowfast.utils.parser import load_config, parse_args


def get_features(video_data, framerate, frame_batch_size=64, model='ResNet-50', device='cuda'):
    """feature extraction"""
    extractor = CNNModel(model=model).to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        args = parse_args()
        cfg = load_config(args)
        num_block = 0
        for fr in range(int(framerate/2),video_length-2,framerate):
            frame_start = max(0, fr-int(frame_batch_size/2))
            frame_end = min(video_length-2, frame_start + frame_batch_size)
            batch = video_data[frame_start:frame_end]
            inputs = process_cv2_inputs(batch, cfg)
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda()
            features_mean, features_std = extractor(inputs)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            num_block = num_block + 1
        output = torch.cat((output1, output2), 1).squeeze()
    if output.ndim == 1:
        output = output.unsqueeze(0)

    return output

if __name__ == "__main__":
    parser = ArgumentParser(description='Extracting Video Motion Features using model-based transfer learning')
    parser.add_argument("--seed", type=int, default=19901116)
    parser.add_argument('--database', default='LIVE_HFR', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='MotionExtractor', type=str,
                        help='which pre-trained model used (default: ResNet-50)')
    parser.add_argument('--frame_batch_size', type=int, default=32,
                        help='frame batch size for feature extraction (default: 64)')
    parser.add_argument('--disable_gpu', action='store_true', help='flag whether to disable GPU')
    parser.add_argument("--ith", type=int, default=150, help='start video id')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNVid':
        videos_dir = '/home/zhengqi/VQA/dataset/KoNViD_1k_videos/'  # videos dir, e.g., ln -s /xxx/KoNViD-1k/ KoNViD-1k
        features_dir = 'CNN_features_KoNViD-1k/MotionFeature_persec_downsample/'  # features dir
        datainfo = 'data/KoNViD.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'LIVE-VQC':
        videos_dir = '/home/zhengqi/VQA/dataset/VideoDatabase/'
        features_dir = 'CNN_features_LIVE-VQC/MotionFeature_persec/'
        datainfo = 'data/LIVE-VQC.mat'
    if args.database == 'YouTube-UGC':
        videos_dir = '/home/zhengqi/VQA/'
        features_dir = 'CNN_features_YouTube-UGC/MotionFeature_persec/'
        datainfo = 'data/YouTube-UGC.mat'
    if args.database == 'LIVE_HFR':
        videos_dir = '/home/zhengqi/VQA/dataset/database/'
        features_dir = 'CNN_features_LIVE_HFR/MotionFeature_persec_downsample/'
        datainfo = 'data/LIVE_HFR.mat'
    if args.database == 'BVI_HFR':
        videos_dir = '/home/zhengqi/VQA/dataset/BVI_HFR_database/'
        features_dir = 'CNN_features_BVI_HFR/MotionFeature_persec_downsample/'
        datainfo = 'data/BVI_HFR_withframerate.mat'

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                  range(len(Info['video_names'][0, :]))]
    # video_names = Info['video_names'][0, :]
    scores = Info['scores'][0, :]
    # video_format = Info['video_format'][()].tobytes()[::2].decode()
    video_format = [Info[Info['video_format'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_format'][0, :]))]
    width = Info['width'][0, :]
    height = Info['height'][0, :]
    # width = Info['width']
    # height = Info['height']
    framerate = Info['framerate'][0, :]
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height, framerate)

    max_len = 0
    min_len = 100000
    complexity_30fps_idx = [155,163, 169]
    #for i in range(args.ith, len(dataset)):
    for i in complexity_30fps_idx:
        current_data = dataset[i]
        print('Video {}: length {}'.format(i, current_data['video'].shape[0]))
        if max_len < current_data['video'].shape[0]:
            max_len = current_data['video'].shape[0]
        if min_len > current_data['video'].shape[0]:
            min_len = current_data['video'].shape[0]
        print(current_data['framerate'])
        start = time.time()
        features = get_features(current_data['video'], int(current_data['framerate']), args.frame_batch_size, args.model, device)
        meanfeat = features.mean(dim=0)
        #np.save(features_dir + str(i) + '_' + args.model + '_last_conv', meanfeat.to('cpu').numpy())
        # np.save(features_dir + str(i) + '_score', current_data['score'])
        end = time.time()
        print('{} seconds'.format(end - start))
    print('Max length: {} Min length: {}'.format(max_len, min_len))

