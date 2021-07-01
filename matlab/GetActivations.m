%  Function gets activations of network (net) when it infers on an image
%  (img), for only specified layers (layers_to_use)
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function activations = GetActivations(img,net,layers_to_use)

    % Check the type of the input image
    if ~isa(img, 'single')
        img = single(img);
    end
    
    % Forward pass
    net.eval({'x0',gpuArray(img)});
    
    % Insert activations inside cell array
    activations = cell(size(layers_to_use,1),2);

    for i=1:size(layers_to_use,1)
        activations{i,1} = layers_to_use{i};
        
        activations{i,2} = ...
        gather(net.vars(getVarIndex(net,net.layers(getLayerIndex(net, layers_to_use{i})).outputs)).value); 
    end

    % Reset network for memory issues     
    reset(net);
    
end