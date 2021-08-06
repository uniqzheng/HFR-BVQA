function feats_frames = calc_HIFRVIQ_features(test_video, width, height, ...
                                            framerate, pixfmt, minside, log_level)
    feats_frames = [];
    % Try to open test_video; if cannot, return
    test_file = fopen(test_video,'r');
    if test_file == -1
        fprintf('Test YUV file not found.');
        feats_frames = [];
        return;
    end
    % Open test video file
    fseek(test_file, 0, 1);
    file_length = ftell(test_file);
    if log_level == 1
        fprintf('Video file size: %d bytes (%d frames)\n',file_length, ...
                floor(file_length/width/height/1.5));
    end
    % get frame number
    if strcmp(pixfmt, 'yuv420p')
        nb_frames = floor(file_length/width/height/1.5); % for 8 bit
    else
        nb_frames = floor(file_length/width/height/3.0); % for 10 bit
    end
    fprintf('nb_frames: %d\n',nb_frames);
    % get features for each chunk
    blk_idx = 0;
    for fr = floor(framerate/2):framerate:nb_frames-2
        blk_idx = blk_idx + 1;
        if log_level == 1
        fprintf('Processing %d-th block...\n', blk_idx);
        end
        % read uniformly sampled 3 frames for each 1-sec chunk
        this_YUV_frame = YUVread(test_file,[width height],fr);
        prev_YUV_frame = YUVread(test_file,[width height],max(1,fr-floor(framerate/3)));
        this_rgb = ycbcr2rgb(uint8(this_YUV_frame));
    	prev_rgb = ycbcr2rgb(uint8(prev_YUV_frame));

        % subsample to 512p resolution
        sside = min(size(this_YUV_frame,1), size(this_YUV_frame,2));
        ratio = minside / sside;
        if ratio < 1
            prev_rgb = imresize(prev_rgb, ratio);
        end
        
        feats_per_frame = [];
        %% extract spatial NSS features - 272-dim
        if log_level == 1
        tic
        fprintf('- Extracting Spatial NSS features (2 fps) ...')
        end
        this_feats_spt = HIFRVIQ_spatial_features(this_rgb);
        if log_level == 1, toc; end
        feats_per_frame = [feats_per_frame,this_feats_spt];
        
        %% extract temporal NSS features - 476-dim
        if log_level == 1
        fprintf('- Extracting temporal NSS features (8 fps) ...')
        tic
        end
        wfun = load(fullfile('include', 'WPT_Filters', 'haar_wpt_3.mat'));
        wfun = wfun.wfun;
        frames_wpt = zeros(size(prev_rgb, 1), size(prev_rgb, 2), size(wfun, 2));
        fr_idx_start = max(1, fr - floor(size(wfun, 2) / 2));
        fr_idx_end = min(nb_frames - 3, fr_idx_start + size(wfun, 2) - 1);
        fr_wpt_cnt = 1;
        % read enough number of frames for temporal bandpass
        for fr_wpt = fr_idx_start:fr_idx_end
            YUV_tmp = YUVread(test_file, [width height], fr_wpt);
            if ratio < 1
                frames_wpt(:,:,fr_wpt_cnt) = imresize(YUV_tmp(:,:,1), ratio);
            else
                frames_wpt(:,:,fr_wpt_cnt) = YUV_tmp(:,:,1);
            end
            fr_wpt_cnt = fr_wpt_cnt + 1;
        end
        dpt_filt_frames = zeros(size(prev_rgb, 1), size(prev_rgb, 2), size(wfun, 1));
        % compute 1D convolution along time (3rd) dimension
        for freq = 1:size(wfun, 1)
            dpt_filt_frames(:,:,freq) = sum(frames_wpt .* ...
                reshape(wfun(freq,:),1,1,[]), 3);
        end
        kscale = 2; % extract at 2 scales
        feats_tmp_wpt = [];
        for ch = 1:size(dpt_filt_frames, 3)
            if ratio < 1
                feat_map = imresize(dpt_filt_frames(:,:,ch), ratio);
            else
                feat_map = dpt_filt_frames(:,:,ch);
            end
            for scale = 1:kscale
                y_scale = imresize(feat_map, 2 ^ (-(scale - 1)));
                feats_tmp_wpt = [feats_tmp_wpt, rapique_basic_extractor(y_scale)];
            end
        end
        if log_level == 1, toc; end
        feats_per_frame = [feats_per_frame, feats_tmp_wpt];
        feats_frames(end+1,:) = feats_per_frame;
    end
    fclose(test_file);
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