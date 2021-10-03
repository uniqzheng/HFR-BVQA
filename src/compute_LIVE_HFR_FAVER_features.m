% Compute features for a set of video files from datasets
% 
close all; 
clear;
addpath(genpath('../include'));

%% parameters
algo_name = 'FAVER_Bior22';  %haar, db2, bior22

data_name = 'LIVE_HFR';
data_path = '../../database';
write_file = true;

video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
out_mat_name = fullfile(feat_path, [data_name,'_',algo_name,'_feats.mat']);
minside = 512.0;
log_level = 0;
feats_mat = zeros( num_videos, 748);

for i = 1:num_videos
    % get video file name
    strs = strsplit(filelist.Filename{i}, '.');
    video_name = fullfile(data_path,filelist.Filename{i});
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    
    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));
  
    feats = [];

	feats = calc_FAVER_features(video_name,width, height, ...
                                    framerate, filelist.pixfmt{i}, minside, log_level);
    feats_mat(i,:) = nanmean(feats);  %求平均
    if write_file
        save(out_mat_name, 'feats_mat');
    end
end

if write_file
    save(out_mat_name, 'feats_mat');
end

