%  Matlab (MatConvNet) implementation of our paper 
%  DFM: A Performance Baseline for Deep Feature Matching 
%  at CVPR 2021 Image Matching Workshop.
% 
%  See details at https://github.com/ufukefe/DFM
%
% @authors: ufukefe, kutalmisince 
% Created on March 23, 2021
% @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function [pointsA, pointsB] = DFM_VGG_stage1(imgA, imgB, model, ratios)
    
    resize_coeff_A = 1;
    resize_coeff_B = 1;
    % Check the image size
    if size(imgA,1)>1600 || size(imgA,2)>1600
        resize_coeff_A = 1600/max(size(imgA,1),size(imgA,2));
        imgA = imresize(imgA,resize_coeff_A);
    end
    
    if size(imgB,1)>1600 || size(imgB,2)>1600
        resize_coeff_B = 1600/max(size(imgB,1),size(imgB,2));
        imgB = imresize(imgB,resize_coeff_B);
    end
    
    size_org_A = size(imgA);
    size_org_B = size(imgB);
    
    % Normalize Images
    imgA = single(imgA);
    imgA = bsxfun(@minus, imgA, model.meta.normalization.averageImage);

    imgB = single(imgB);
    imgB = bsxfun(@minus, imgB, model.meta.normalization.averageImage);
    
    % zero padding for vgg (canvas should be a multiple of 16)
    imgA = ZeroPadding4VGG(imgA);
    imgB = ZeroPadding4VGG(imgB);
    
    % Assign layers to be used
    layers_to_use_A = {'conv5_2';'conv4_2';'conv3_2';'conv2_2';'conv1_2'};
    layers_to_use_B = {'conv5_2';'conv4_2';'conv3_2';'conv2_2';'conv1_2'};
    model = ArrangeNetwork(model,layers_to_use_A);
    
    % Assign ratio tests for layers
    ratios_s1 = ratios(5:-1:1);
    
    % get activations
    activationsA = GetActivations(imgA, model, layers_to_use_A);
    activationsB = GetActivations(imgB, model, layers_to_use_B);

    % initiate matches
    [pointsA, pointsB] = DenseFeatureMatching(activationsA{1,2}, activationsB{1,2},ratios_s1(1));  
    [pointsA, pointsB] = insideImage(pointsA,pointsB,size_org_A,size_org_B,16);
    
    for k = 2:5
        [pointsA, pointsB] = RefinePoints(pointsA, pointsB, activationsA{k,2}, activationsB{k,2}, ratios_s1(k));
    end
    
    % Reject matches at the side of the images
    [pointsA,pointsB] = rejectSideMatches(pointsA,pointsB,size_org_A,size_org_B,16);    
    
    if resize_coeff_A ~= 1
       pointsA = (1/resize_coeff_A)*(pointsA-0.5)+0.5;
    end
    
    if resize_coeff_B ~= 1
       pointsB = (1/resize_coeff_B)*(pointsB-0.5)+0.5;
    end
    
end
