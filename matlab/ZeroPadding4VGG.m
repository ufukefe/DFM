%  Function zero pads images as their sides become product of 16 to
%  make VGG Network work smoothly.
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function imgOut = ZeroPadding4VGG(imgIn)

    [h, w, ch] = size(imgIn);
    
    if mod(h, 16) || mod(w, 16)
        imgOut = single(zeros(ceil(h/ 16) * 16, ceil(w/ 16) * 16, ch));
        imgOut(1:h, 1:w, 1:ch) = imgIn;
    else
        imgOut = imgIn;
        
    end
end

        