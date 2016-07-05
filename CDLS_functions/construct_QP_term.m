% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)

function [global_MMD_struct, local_MMD_struct, pair_struct] = construct_QP_term(X_S,X_L,X_U,S_Label,L_Label,U_Label,delta)

%%%%% Parameter Setting %%%%%
ns = size(S_Label,1);
nl = size(L_Label,1);
nu = size(U_Label,1);

%%%%% Number of Categories %%%%%
lbl_idx = unique(S_Label);
C = length(lbl_idx);

%%%%% Kernelize %%%%%
ker = 'linear';
gamma = 1; % no use here
K = kernel(ker,[X_S,X_L,X_U],[],gamma);

%%%%% Construct global_MMD_struct %%%%%
a0 = 1/(ns*delta);
b0 = 1/(nu*delta+nl);
global_MMD_struct.between_alpha_alpha = a0*a0*K(1:ns,1:ns);
global_MMD_struct.between_beta_beta = a0*b0*K(ns+nl+1:end,ns+nl+1:end);
global_MMD_struct.between_beta_alpha = -a0*b0*K(ns+nl+1:end,1:ns);
global_MMD_struct.before_alpha = -a0*b0*sum(K(1:ns,ns+1:ns+nl),2);
global_MMD_struct.before_beta = b0*b0*sum(K(ns+nl+1:end,ns+1:ns+nl),2);

%%%%% Construct local_MMD_struct %%%%%
mul_mtx = zeros(ns+nl+nu,ns+nl+nu);
for i = 1:C
    ac = 1/(sum(S_Label==lbl_idx(i))*delta);
    bc = 1/(sum(U_Label==lbl_idx(i))*delta+sum(L_Label==lbl_idx(i)));
    
    e = zeros(ns+nl+nu,1);
    e(S_Label==lbl_idx(i)) = ac*ones(length(find(S_Label==lbl_idx(i))),1);
    e(ns+find(L_Label==lbl_idx(i))) = bc*ones(length(find(L_Label==lbl_idx(i))),1);
    e(ns+nl+find(U_Label==lbl_idx(i))) = bc*ones(length(find(U_Label==lbl_idx(i))),1);
    
    mul_mtx = mul_mtx + e*e';
end
Kpron = mul_mtx.*K;

local_MMD_struct.between_alpha_alpah = Kpron(1:ns,1:ns);
local_MMD_struct.between_beta_beta = Kpron(ns+nl+1:end,ns+nl+1:end);
local_MMD_struct.between_beta_alpha = -Kpron(ns+nl+1:end,1:ns);
local_MMD_struct.before_alpha = -sum(Kpron(1:ns,ns+1:ns+nl),2);
local_MMD_struct.before_beta = sum(Kpron(ns+nl+1:end,ns+1:ns+nl),2);

%%%%% Construct pair_struct %%%%%

mul_mtx = zeros(ns+nl+nu,ns+nl+nu);
for i = 1:C
    tp1 = sum(S_Label==lbl_idx(i))*delta;
    tp2 = sum(U_Label==lbl_idx(i))*delta;
    nlc = sum(L_Label==lbl_idx(i));
    ec = 1/(tp1*tp2+tp2*nlc+nlc*tp1);
    
    e = zeros(ns+nl+nu,1);
    e(S_Label==lbl_idx(i)) = ones(length(find(S_Label==lbl_idx(i))),1);
    e(ns+find(L_Label==lbl_idx(i))) = ones(length(find(L_Label==lbl_idx(i))),1);
    e(ns+nl+find(U_Label==lbl_idx(i))) = ones(length(find(U_Label==lbl_idx(i))),1);
    
    mul_mtx = mul_mtx + ec*(e*e');
end
Kppron = mul_mtx.*K;

tmpsl = repmat(diag(Kppron(1:ns,1:ns)),1,nl);
tmplls = repmat(diag(Kppron(ns+1:ns+nl,ns+1:ns+nl))',ns,1);
tmp1 = 0.5*(tmpsl+tmplls);

tmpul = repmat(diag(Kppron(ns+nl+1:end,ns+nl+1:end)),1,nl);
tmpllu = repmat(diag(Kppron(ns+1:ns+nl,ns+1:ns+nl))',nu,1);
tmp2 = 0.5*(tmpul+tmpllu);

tmpus = repmat(diag(Kppron(ns+nl+1:end,ns+nl+1:end)),1,ns);
tmpssu  = repmat(diag(Kppron(1:ns,1:ns))',nu,1);
tmp3 = 0.5*(tmpus+tmpssu);

pair_struct.between_beta_alpha = tmp3-Kppron(ns+nl+1:end,1:ns);
pair_struct.before_alpha = sum(tmp1,2)-sum(Kppron(1:ns,ns+1:ns+nl),2);
pair_struct.before_beta = sum(tmp2,2)-sum(Kppron(ns+nl+1:end,ns+1:ns+nl),2);


