%  Function collects HPatches evaluation results in terms of MMA, 
%  Homography Estimation and Number of Matches.
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle east technical university, center for image analysis
%  Last Edited on July 1, 2021

function results = get_hpatches_results(pointsA,pointsB,h_gt,sizeA)   
    if isempty(pointsA)
            mma = zeros(1,10);
            num_points = 0;
            hqual_max = zeros(1,5);
            hqual_min = zeros(1,5);
            hqual_all = zeros(10,5);
            
     elseif 1<=size(pointsA,1) && size(pointsA,1)<4
               
        mma = calculateMMA (pointsA, pointsB, h_gt);
        num_points = size(pointsA,1);
        hqual_max = zeros(1,5);
        hqual_min = zeros(1,5);
        hqual_all = zeros(10,5);
            
    else
        mma = calculateMMA (pointsA, pointsB, h_gt);
        num_points = size(pointsA,1);
        
    %        Look for 10 tests 
        hqual_all = [];
        for i = 1:10
            hqual = calculateHOMquality(pointsA, pointsB, sizeA, h_gt);
            hqual_all = [hqual_all;hqual];
            hqual_all_total = sum(hqual_all,2);
            [~,idx] = max(hqual_all_total);
            hqual_max = hqual_all(idx,:);
            [~,idx] = min(hqual_all_total);
            hqual_min = hqual_all(idx,:);
        end        
    end
    results = [mma num_points hqual_max hqual_min reshape(hqual_all',[1 50])];
end