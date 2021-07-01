%  Function performs Dense Nearest Neighbour Search which represented in
%  our paper DFM: A Performance Baseline for Deep Feature Matching. For
%  more details, see:
%  https://openaccess.thecvf.com/content/CVPR2021W/IMW/papers/Efe_DFM_A_Performance_Baseline_for_Deep_Feature_Matching_CVPRW_2021_paper.pdf
%
%  This funcion can be considered as "Mutual Nearest Neighbor Search with
%  ratio test".
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021


function [points_A, points_B, matchmetrics] = DenseFeatureMatching(acts_A, acts_B,ratio_threshold)
    
    % get the size of A, B
    [h_A, w_A, ~] = size(acts_A);
    [h_B, w_B, ~] = size(acts_B);
    
    % form the coordinate grid for A, B
    X_A = ones(h_A, 1) * (1:w_A);
    Y_A = (1:h_A)' * ones(1,w_A);
    
    X_B = ones(h_B, 1) * (1:w_B);
    Y_B = (1:h_B)' * ones(1,w_B);
    
    % form points for A, B
    P_A = [X_A(:), Y_A(:)];
    P_B = [X_B(:), Y_B(:)];
    
    % reshape and normalize activations    
    norms_A = sqrt(sum(acts_A.^2,3));
    acts_A = bsxfun(@rdivide, acts_A, norms_A);
    
    norms_B = sqrt(sum(acts_B.^2,3));
    acts_B = bsxfun(@rdivide, acts_B, norms_B);
    
    % find matches
    [matches, matchmetrics] = matchFeatures(reshape(acts_A,[size(P_A,1),size(acts_A,3)]),...
        reshape(acts_B,[size(P_B,1),size(acts_B,3)]),...
        'Unique',1,'MaxRatio',ratio_threshold,'MatchThreshold',100,'Metric','SSD');

    points_A = P_A(matches(:,1),:);
    points_B = P_B(matches(:,2),:);
%     numP = size(matches,1);
    
end