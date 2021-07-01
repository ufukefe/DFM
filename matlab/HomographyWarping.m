%  Function warps given image (B) onto given plane (sizeA) using
%  homography matrix (H) via bilinear interpolation and returns warped image
%  (B_w) 
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle east technical university, center for image analysis
%  Edited on March 24, 2021
%  Last Edited on July 1, 2021

function B_w = HomographyWarping(B, sizeA, H)

    % convert to double
    B = double(B);
    
    % get the size of B
    sizeB = size(B);

    % form the coordinate grid for A
    X = ones(sizeA(1), 1) * (1:sizeA(2));
    Y = (1:sizeA(1))' * ones(1,sizeA(2));
    
    % form points for transformation from A to B
    P = [X(:) - 0.5, Y(:) - 0.5, ones(sizeA(2) * sizeA(1), 1)]';

    % transform points with homography
    P_ = H * P;

    % find coordinates on B
    x = P_(1, :) ./ P_(3, :) + 0.5;
    y = P_(2, :) ./ P_(3, :) + 0.5;

    % find the point indexes lying in the canvas of B
    in_image = and(and(floor(x) >= 1, floor(x) < sizeB(2)), and(floor(y) >= 1, floor(y) < sizeB(1)));

    % set 1D index of the grid on B and Bw
    indA = Y(:) + (X(:) - 1) * sizeA(1);
    indA = indA(in_image);
    
    indB = floor(y(in_image)) + (floor(x(in_image)) - 1) * sizeB(1);

    % find the folating part of x and y
    xF = x(in_image) - floor(x(in_image));
    yF = y(in_image) - floor(y(in_image));
    
    % allocate space for the warped image
    B_w = zeros(sizeA, 'single');

    % fill the warped image with bilinear interpolation
    for c = 1 : sizeA(3)
        B_w(indA + (c - 1) * sizeA(2) * sizeA(1)) = B(indB + (c - 1) * sizeB(2) * sizeB(1)) .* (1 - xF) .* (1 - yF) ...
                                                  + B(indB + (c - 1) * sizeB(2) * sizeB(1) + 1) .* (1 - xF) .* yF ...
                                                  + B(indB + (c - 1) * sizeB(2) * sizeB(1) + sizeB(1)) .* xF .* (1 - yF) ...
                                                  + B(indB + (c - 1) * sizeB(2) * sizeB(1) + sizeB(1) + 1) .* xF .* yF;
    end
end
