% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)

function [alpha,beta] = QP_choose_source_unlabel(X_S,X_L,X_U,S_Label,L_Label,U_Label,delta)

%%%%% Construct QP Term %%%%%
[global_MMD_struct, local_MMD_struct, pair_struct] = construct_QP_term(X_S,X_L,X_U,S_Label,L_Label,U_Label,delta);

%%%%% input of QP %%%%%
Kss = global_MMD_struct.between_alpha_alpha + local_MMD_struct.between_alpha_alpah;
Kus = global_MMD_struct.between_beta_alpha + local_MMD_struct.between_beta_alpha + pair_struct.between_beta_alpha;
Kuu = global_MMD_struct.between_beta_beta + local_MMD_struct.between_beta_beta;
ksl = global_MMD_struct.before_alpha + local_MMD_struct.before_alpha + pair_struct.before_alpha;
kul = global_MMD_struct.before_beta + local_MMD_struct.before_beta + pair_struct.before_beta;

H = [Kss,Kus';Kus,Kuu];
f = [ksl;kul];
A = [];
b = [];
Aeq_alpha = OneOfKEncoding(S_Label)';
Aeq_beta = OneOfKEncoding(U_Label)';
Aeq = [Aeq_alpha,zeros(size(Aeq_alpha,1),length(U_Label));zeros(size(Aeq_beta,1),length(S_Label)),Aeq_beta];
beq_alpha = delta*(sum(Aeq_alpha,2));
beq_beta = delta*(sum(Aeq_beta,2));
beq = [beq_alpha;beq_beta];
lb = zeros(length(S_Label)+length(U_Label),1);
ub = ones(length(S_Label)+length(U_Label),1);

options = optimset('Display','off', 'MaxIter', 1500, 'Algorithm', 'interior-point-convex','TolFun',1e-15);      
alpha_beta = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

alpha = alpha_beta(1:length(S_Label));
beta = alpha_beta(length(S_Label)+1:end);

