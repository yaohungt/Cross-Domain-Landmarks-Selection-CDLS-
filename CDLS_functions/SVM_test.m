% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)

function [acc,largest_idx] = SVM_test(Cs,alpha,co_Xs,S_Label,co_Xt,T_Label,co_Xtest,Ttest_Label)

alpha = Cs*alpha;

pivot_label = [S_Label;T_Label];
pivot_feature = [co_Xs;co_Xt];
pivot_weight = [alpha;ones(length(T_Label),1)];

model = svmtrain(pivot_weight,pivot_label,sparse(pivot_feature),'-s 1 -t 0 -q');
[largest_idx,acc,~] = svmpredict(Ttest_Label,sparse(co_Xtest),model,'-q');
acc = acc(1);
