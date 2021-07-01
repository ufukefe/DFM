%  Function arranges given network (net) according to given layers to use
%  (layers_to_use), and outputs the arranged nework.
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

function net = ArrangeNetwork(net,layers_to_use)

    % Loads the network into the wrapper
    net = dagnn.DagNN.loadobj(net);    

    % Set network to test model.
    net.mode = 'test';

    % Remove unnecessary layers
    net.removeLayer({net.layers(getLayerIndex(net, layers_to_use{1})+1:size(net.layers,2)).name}) 

    % Remove unused activations from RAM
    net.conserveMemory = 1;                                

    % Keep activations that will be used
    for i=1:size(layers_to_use,1)   
    net.vars(getVarIndex(net,net.layers(getLayerIndex(net, layers_to_use{i})).outputs)).precious = 1;
    end

    % Move network to gpu         
    net.move('gpu');

end