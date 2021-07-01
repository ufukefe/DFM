%  Function performs Hiearchical Refinement which represented in
%  our paper DFM: A Performance Baseline for Deep Feature Matching. For
%  more details, see:
%  https://openaccess.thecvf.com/content/CVPR2021W/IMW/papers/Efe_DFM_A_Performance_Baseline_for_Deep_Feature_Matching_CVPRW_2021_paper.pdf
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function [pointsRA, pointsRB] = RefinePoints(pointsA, pointsB, acts_A, acts_B, ratio_threshold)

    % initate number of points
    numInputPoints = size(pointsA, 1);
    numOutputPoints = 0;
    
    % allocate space refined points
    pointsRA = zeros(numInputPoints * 4, 2);
    pointsRB = zeros(numInputPoints * 4, 2);
    
    % upsample points by 2
    pointsA_RB = pointsA * 2; pointsA_LT = pointsA * 2 - 1;
    pointsB_RB = pointsB * 2; pointsB_LT = pointsB * 2 - 1;
    
    % for each point search the refined matches in given (finer) resolution
    for n = 1 : numInputPoints
        
        actA = acts_A(pointsA_LT(n,2):pointsA_RB(n,2), pointsA_LT(n,1):pointsA_RB(n,1), :);
        actB = acts_B(pointsB_LT(n,2):pointsB_RB(n,2), pointsB_LT(n,1):pointsB_RB(n,1), :);
        
        [pA, pB, matchmetrics] = DenseFeatureMatching(actA, actB, ratio_threshold);
        
     % Get best 2 matches
        if size(matchmetrics,1)>=2
            [~,idx] = sort(matchmetrics);
            pA = pA(idx,:);
            pB = pB(idx,:);
            
            pA = pA(1:2,:);
            pB = pB(1:2,:);
        end
        
        numP = size(pA,1);
              
        pointsRA(numOutputPoints + 1 : numOutputPoints + numP, :) = pA + pointsA_LT(n,:) - 1;
        pointsRB(numOutputPoints + 1 : numOutputPoints + numP, :) = pB + pointsB_LT(n,:) - 1;
        
        numOutputPoints = numOutputPoints + numP;
    end
    pointsRA = pointsRA(1:numOutputPoints,:);
    pointsRB = pointsRB(1:numOutputPoints,:);
end