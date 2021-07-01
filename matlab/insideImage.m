%  Function checks whether points inside the image or not, and returns
%  points inside the image. This funcion is necessary because we zero
%  padded images as their sides become product of 16.
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021


function [points_A_out, points_B_out] = insideImage(points_A_in,points_B_in,size_A,size_B,resolution)
    
    points_A_in = (points_A_in - 0.5) * resolution + 0.5; 
    points_B_in = (points_B_in - 0.5) * resolution + 0.5; 

    h_A = size_A(1);
    w_A = size_A(2); 
    
    h_B = size_B(1);
    w_B = size_B(2);
    
    ind_A = (1:size(points_A_in,1))';
  
    % Check all points on A inside image
    in_image_A = and(and(points_A_in(:,1) >= 1, points_A_in(:,1) <= w_A),...
        and(points_A_in(:,2) >= 1, points_A_in(:,2) <= h_A));
    
    ind_A = ind_A(in_image_A);
    
    % Keep points inside image A
    points_A_out = points_A_in(ind_A,:);
    points_B_out = points_B_in(ind_A,:);
        
    ind_B = (1:size(points_B_out,1))';
    
    % Check all points on B inside image
    in_image_B = and(and(points_B_out(:,1) >= 1, points_B_out(:,1) <= w_B),...
        and(points_B_out(:,2) >= 1, points_B_out(:,2) <= h_B));
    
    ind_B = ind_B(in_image_B);
    
    % Keep points inside image A
    points_A_out = points_A_out(ind_B,:);
    points_B_out = points_B_out(ind_B,:);
    
    points_A_out = (points_A_out - 0.5) / resolution + 0.5; 
    points_B_out = (points_B_out - 0.5) / resolution + 0.5; 
end