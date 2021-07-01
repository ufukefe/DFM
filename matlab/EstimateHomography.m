%  Function estimates homography for given two point sets (points_A and
%  points_B)
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function homography = EstimateHomography(points_A, points_B)

    % Estimates homography between two point sets
    [homography,~,~] = estimateGeometricTransform(points_A,points_B,...
    'projective','Confidence',99.99,'MaxNumTrials',5000,'MaxDistance',16*sqrt(2)+1);

end