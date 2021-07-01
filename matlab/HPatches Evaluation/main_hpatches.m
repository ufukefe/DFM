%  HPatches evaluation of Matlab (MatConvNet) implementation of our paper 
%  DFM: A Performance Baseline for Deep Feature Matching 
%  at CVPR 2021 Image Matching Workshop.
% 
%  See details at https://github.com/ufukefe/DFM
%
%  @authors: ufukefe, kutalmisince 
%  Created on March 23, 2021
%  @Middle East Technical University, Center for Image Analysis
%  Last Edited on July 1, 2021

%% Clear variables
clear all;
close all;
clc;

%% Get Inputs

% Load the trained model 
[parentdir,~,~]=fileparts(pwd);
addpath(parentdir)

model = load([parentdir, '\models\imagenet-vgg-verydeep-19.mat']);

% Dataset name and result file
dataset = [parentdir, '\data\hpatches-sequences-release\hpatches-sequences-release\'];
result_dir = [parentdir, '\results\HPatches\'];

if ~(isfolder(result_dir))
mkdir (result_dir)
end

image_dir = strcat(dataset);
image_files = dir(fullfile(image_dir,'*'));
image_files = image_files(~ismember({image_files(:).name},{'.','..'}));

% Parameters
ratios = [0.6, 0.6, 0.8, 0.9, 0.95, 1.0];

% To write txt
fileID = fopen(strcat(result_dir,'results.txt'),'a');

for i = 1:size(image_files,1)
    %     Get inputs
    images = dir(fullfile(strcat(dataset,image_files(i).name),'*ppm'));
    homographies = dir(fullfile(strcat(dataset,image_files(i).name),'*'));
    homographies = homographies(9:13);
     
    %     Start experiments
    imgA = imread(strcat(images(1).folder,'\',images(1).name));
        
    for k=1:size(homographies,1)
        imgB = imread(strcat(images(k+1).folder,'\',images(k+1).name));
        [pointsA, pointsB] = DFM_VGG(imgA, imgB, model, ratios);
        
        h_gt = load(strcat(homographies(k).folder,'\',homographies(k).name));
        
        results = get_hpatches_results(pointsA,pointsB,h_gt,size(imgA));
        fprintf(fileID,strcat('%s', '\n'),...
            [image_files(i).name,'_',num2str(1),'_',num2str(k+1), ' ', num2str(results)]);
    end
end

% Close txt
fclose(fileID);   



    


