%  Function calculates Mean Matching Accuracy (MMA).
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle east technical university, center for image analysis
%  Last Edited on July 1, 2021

function mma = calculateMMA(pointsA, pointsB, h_gt)


    pointsB_gt = h_gt * [pointsA - 0.5, ones(size(pointsA, 1),1)]';

    x = pointsB_gt(1, :) ./ pointsB_gt(3, :) + 0.5;
    y = pointsB_gt(2, :) ./ pointsB_gt(3, :) + 0.5;

    pointsB_gt = [x; y];
    pointsB_gt = pointsB_gt';
    
    distances = sqrt(sum((pointsB_gt - pointsB).^2,2));
    
    mma = zeros(1,10);

    for th = 1 : 10
        mma(th) = sum(distances <= th);
    end

    mma = mma / size(distances, 1);
    
end
