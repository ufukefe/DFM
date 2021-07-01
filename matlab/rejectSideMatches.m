%  Function rejets matches that exist at near edges. These matches may be
%  noisy, therefore we delete them. However using this function made reduce the
%  Homography Estimation performance.
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function [points_A_out, points_B_out] = rejectSideMatches(points_A_in,points_B_in,size_A,size_B,reject_area)

    h_A = size_A(1);
    w_A = size_A(2); 
    
    h_B = size_B(1);
    w_B = size_B(2);
    
    ind_A = (1:size(points_A_in,1))';
  
    % Check all points inside image but not inside rejection area
    in_image_A = and(and(points_A_in(:,1) >= reject_area+1, points_A_in(:,1) <= w_A-reject_area),...
        and(points_A_in(:,2) >= reject_area+1, points_A_in(:,2) <= h_A-reject_area));
    
    ind_A = ind_A(in_image_A);
    
    % Keep points inside image A
    points_A_out = points_A_in(ind_A,:);
    points_B_out = points_B_in(ind_A,:);
        
    ind_B = (1:size(points_B_out,1))';
    
    % Check all points on B inside image
    in_image_B = and(and(points_B_out(:,1) >= reject_area+1, points_B_out(:,1) <= w_B-reject_area),...
        and(points_B_out(:,2) >= reject_area+1, points_B_out(:,2) <= h_B-reject_area));
    
    ind_B = ind_B(in_image_B);
    
    % Keep points inside image A
    points_A_out = points_A_out(ind_B,:);
    points_B_out = points_B_out(ind_B,:);
    
end