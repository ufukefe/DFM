%  Function calculates homography estimation quality.
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle east technical university, center for image analysis
%  Last Edited on July 1, 2021

function homq = calculateHOMquality (points_A,points_B,size_A,h_gt)
    
    w = size_A(2);
    h = size_A(1);
    
    % Estimates homography between two point sets
    try
    [h_est,~,~] = estimateGeometricTransform(points_A,points_B,...
    'projective','Confidence',99.99,'MaxNumTrials',5000,'MaxDistance',3);
    h_est = h_est.T';
    catch
    homq = zeros(1,5);    
    return
    end
    
    %     Assign four corners of the first image
    cornersA = [[1,1;1 h;w 1;w h] - 0.5, ones(4,1)]';

    %     Find corners with estimated homography
    cornersA_est = h_est * cornersA;

    x = cornersA_est(1, :) ./ cornersA_est(3, :) + 0.5;
    y = cornersA_est(2, :) ./ cornersA_est(3, :) + 0.5;

    cornersA_est = [x; y];
    cornersA_est = cornersA_est';

    %     Find corners with groundtruth homography
    cornersA_gt = h_gt * cornersA;

    x = cornersA_gt(1, :) ./ cornersA_gt(3, :) + 0.5;
    y = cornersA_gt(2, :) ./ cornersA_gt(3, :) + 0.5;

    cornersA_gt = [x; y];
    cornersA_gt = cornersA_gt';
    
    %     Find distances
    distances = sqrt(sum((cornersA_gt - cornersA_est).^2,2));
    distance = mean(distances);
    
    homq = zeros(1,5);

    for th = 1 : 5
        homq(th) =  distance <= th;
    end


end