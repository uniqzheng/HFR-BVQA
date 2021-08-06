% Compute features for a set of video files from datasets
% 
close all; 
clear;
addpath(genpath('../include/'));


%% parameters
algo_name = 'HIFRVIQ_Haar';
data_name = 'LIVE_HFR';
data_path = '../../database/database';
write_file = true;

video_tmp = '../tmp';
if ~exist(video_tmp, 'dir'), mkdir(video_tmp); end
feat_path = '../features';
mos_filename = fullfile(feat_path, [data_name,'_metadata.csv']);
filelist = readtable(mos_filename);
num_videos = size(filelist,1);
out_mat_name = fullfile(feat_path, [data_name,'_',algo_name,'_feats.mat']);
feats_mat = zeros(num_videos, 748);

tic
for i = 1:num_videos
    % get video file name
    strs = strsplit(filelist.Filename{i}, '_');
    video_name = fullfile(data_path,strs{1},[filelist.Filename{i},'.webm']);
    fprintf('Computing features for %d sequence: %s\n', i, video_name);
    
    % get video meta data
    width = filelist.width(i);
    height = filelist.height(i);
    framerate = round(filelist.framerate(i));
    
    % decode video and store in video_tmp dir
    yuv_name = fullfile(video_tmp, [filelist.Filename{i}, '.yuv']);
    cmd = ['ffmpeg -loglevel error -y -r ', num2str(framerate), ...
        ' -i ', video_name, ' -pix_fmt ', filelist.pixfmt{i},  ...
       ' -s ', [num2str(width),'x',num2str(height)], ' -vsync 0 ', yuv_name];
    system(cmd);

    minside = 512.0;
    log_level = 0;  % 1=verbose, 0=quite

    feats = [];
    feats = calc_HIFRVIQ_features(yuv_name, width, height, ...
                                            framerate, filelist.pixfmt{i}, minside, log_level);
  
    delete(yuv_name)
    feats_mat(i,:) = nanmean(feats);  %求平均
    if write_file
        save(out_mat_name, 'feats_mat');
    end
end
toc
if write_file
    save(out_mat_name, 'feats_mat');
end

% Read one frame from YUV file
function YUV = YUVread(f,dim,frnum,type)

    % This function reads a frame #frnum (0..n-1) from YUV file into an
    % 3D array with Y, U and V components
    if strcmp(type, 'yuv420p')
        %% Start a file pointer
        fseek(f,(frnum-1)*1.5*dim(1)*dim(2), 'bof'); % Frame read for 8 bit ; bof == -1
        %Read Y-component
        Y=fread(f,dim(1)*dim(2),'uchar');
        % Read U-component
        U=fread(f,dim(1)*dim(2)/4,'uchar');    
		% Read V-component
        V=fread(f,dim(1)*dim(2)/4,'uchar');
    else
        fseek(f,(frnum-1)*3.0*dim(1)*dim(2), 'bof'); % Frame read for 10 bit
		%Read Y-component
        Y=fread(f,dim(1)*dim(2),'uint16');
        % Read U-component
        U=fread(f,dim(1)*dim(2)/4,'uint16');    
		% Read V-component
        V=fread(f,dim(1)*dim(2)/4,'uint16');
    end
    %fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    
    % Read Y-component
    if length(Y)<dim(1)*dim(2)
        YUV = [];
        return;
    end
    Y=cast(reshape(Y,dim(1),dim(2)),'double');
	
    % Read U-component
    if length(U)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double');
    U=imresize(U,2.0);
    
    % Read V-component
    if length(V)<dim(1)*dim(2)/4
        YUV = [];
        return;
    end    
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double');
    V=imresize(V,2.0);
    
    % Combine Y, U, and V
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
end
