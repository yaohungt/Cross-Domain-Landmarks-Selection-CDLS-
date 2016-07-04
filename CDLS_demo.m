% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)
%
% Demo script of Cross-Domain Landmarks Selection (CDLS)

clear;clc

%%%%% addpath %%%%%
addpath('./CDLS_functions/');
addpath('./libsvm-weights-3.20/matlab');

%%%%% Data Loading and Preprocessing %%%%%
load('./data/amazon_DeCAF_dslr_SURF.mat');
S = S ./ repmat(sqrt(sum(S.^2,2)),1,size(S,2));
T = T ./ repmat(sqrt(sum(T.^2,2)),1,size(T,2));
Ttest = Ttest ./ repmat(sqrt(sum(Ttest.^2,2)),1,size(Ttest,2));

Data.T = T';
Data.Ttest = Ttest';
Data.S = S';
Data.T_Label = T_Label;
Data.S_Label = S_Label;
Data.Ttest_Label = Ttest_Label;

%%%%% Parameter Setting %%%%%
param.iter = 5;
param.scale = num_of_T_per_class/num_of_L_per_class;
param.delta = 0.5; %% You can tune the portion of the weights if you like (0 < delta <= 1 )
param.PCA_dimension = 100; %% Make sure this dim. is smaller the source-domain dim.

%%%%% Start CDLS %%%%%
clearvars -except Data param
fprintf('Transfering knowledge from Amazon images with DeCAF features to DSLR images with SURF features ...\n');
acc = CDLS(Data,param);
clear Data param