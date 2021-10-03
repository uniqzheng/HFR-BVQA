function feats = FAVER_spatial_features(YUV)
    feats = [];
    kscale = 2;
    kband = 4; % actually 4 bandpass + 1 lowpass
    
    %% parameters got from IL-NIQE
    sigmaForGauDerivative = 1.66;
    scaleFactorForGaussianDer = 0.28;
    Y = YUV(:,:,1);
    U = YUV(:,:,2);
    V = YUV(:,:,3);
    %% Compute feature maps
    % GM
    [GM, ~] = imgradient(Y);
    % LoG
    window2 = fspecial('log', 9, 9/6);
    window2 =  window2/sum(abs(window2(:)));
    LOG = abs(imfilter(Y, window2, 'replicate'));

    % spatial luma features: Y, GM, LOG, DOG1
    compositeMat = [];
    compositeMat = cat(3, compositeMat, Y);
    compositeMat = cat(3, compositeMat, U);
    compositeMat = cat(3, compositeMat, V);
    compositeMat = cat(3, compositeMat, GM);
    compositeMat = cat(3, compositeMat, LOG);

   
    %% calculate spatial features from feature maps
    for ch = 1:size(compositeMat,3)
        for scale = 1:kscale
            % chroma features only in half scale
            if (ch >= 4) && (scale == 1) 
                continue;  
            end
            y_scale = imresize(compositeMat(:,:,ch), 2 ^ (-(scale - 1)));
            feats = [feats rapique_basic_extractor(y_scale)];
        end
    end

end

