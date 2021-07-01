%  Matlab (MatConvNet) implementation of our paper 
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

%% Inputs and Parameters
model = load('models\imagenet-vgg-verydeep-19.mat');

imgA = imread('data\v_adam_1.ppm');
imgB = imread('data\v_adam_3.ppm');

ratios = [0.6, 0.6, 0.8, 0.9, 0.95, 1.0];

%% Run DFM

[pointsA, pointsB] = DFM_VGG(imgA, imgB, model, ratios);

%% Visualization
figure();
showMatchedFeatures(imgA, imgB, pointsA, pointsB, 'montage');