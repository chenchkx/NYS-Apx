%% An implementation of nystrom method for approximating infinite-dimensional sample in the feature space.
% Written by Kai-Xuan Chen, (e-mail: kaixuan_chen_jsh@163.com)
% If you find any bugs, please contact me.
% Date: 2020.06.13


clear;
clc;
data = rand(50,10000);

% input:
%     X: columns of vectors of data points. 
%     num_sampled: the number of sampled data points
%     r: the dimension of the approximate data points
% output
%     X_appro: the approximation of the data points in the kernel space.   
data_approximation = compute_nyApproximation(data,100,99);


