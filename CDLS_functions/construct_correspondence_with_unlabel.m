% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)

function [Hs1,Hs2,Hs3,Hsl1,Hsl2,Hsl3,Hsu1,Hsu2,Hsu3] = construct_correspondence_with_unlabel(S_Label,L_Label,U_Label,alpha,beta)

%%%%% Number of Categories %%%%%
lbl_idx = unique(S_Label);
C = length(lbl_idx);

%%%%% number of instances %%%%%
ns = length(S_Label);
nl = length(L_Label);
nu = length(U_Label);

%%%%% construct Hs1 %%%%%
a0 = 1/sum(alpha);
Hs1 = a0*a0*(alpha*alpha');

%%%%% construct Hs2 %%%%%
Hs2 = zeros(ns,ns);
for i = 1:C
    ac = 1/sum(alpha(S_Label==lbl_idx(i)));
    alpha_c = alpha(S_Label==lbl_idx(i));
    Hs2(S_Label==lbl_idx(i),S_Label==lbl_idx(i)) = ac*ac*(alpha_c*alpha_c');
end

%%%%% construct Hs3 %%%%%
Hs3 = zeros(ns,ns);
%alpha_square = alpha.*alpha;
for i = 1:C
    tmp1 = sum(alpha(S_Label==lbl_idx(i)));
    tmp2 = sum(beta(U_Label==lbl_idx(i)));
    nlc = sum(L_Label==lbl_idx(i));
    ec = 1/(tmp1*tmp2+tmp2*nlc+nlc*tmp1);
    %nuc = sum(U_Label==lbl_idx(i));
    %Hs3(S_Label==lbl_idx(i),S_Label==lbl_idx(i)) = (ec*(nuc+nlc))*diag(alpha_square(S_Label==lbl_idx(i)));
    Hs3(S_Label==lbl_idx(i),S_Label==lbl_idx(i)) = (ec*(tmp2+nlc))*diag(alpha(S_Label==lbl_idx(i)));
end

%%%%% construct Hsl1 %%%%%
a0 = 1/sum(alpha);
b0 = 1/(sum(beta)+nl);
Hsl1 = a0*b0*(alpha*ones(1,nl));

%%%%% construct Hsl2 %%%%%
Hsl2 = zeros(ns,nl);
for i = 1:C
    ac = 1/sum(alpha(S_Label==lbl_idx(i)));
    bc = 1/(sum(beta(U_Label==lbl_idx(i)))+sum(L_Label==lbl_idx(i)));
    alpha_c = alpha(S_Label==lbl_idx(i));
    Hsl2(S_Label==lbl_idx(i),L_Label==lbl_idx(i)) = ac*bc*(alpha_c*ones(1,sum(L_Label==lbl_idx(i))));
end

%%%%% construct Hsl3 %%%%%
Hsl3 = zeros(ns,nl);
for i = 1:C
    tmp1 = sum(alpha(S_Label==lbl_idx(i)));
    tmp2 = sum(beta(U_Label==lbl_idx(i)));
    nlc = sum(L_Label==lbl_idx(i));
    ec = 1/(tmp1*tmp2+tmp2*nlc+nlc*tmp1);
    alpha_c = alpha(S_Label==lbl_idx(i));
    Hsl3(S_Label==lbl_idx(i),L_Label==lbl_idx(i)) = ec*(alpha_c*ones(1,sum(L_Label==lbl_idx(i))));
end

%%%%% construct Hsu1 %%%%%
a0 = 1/sum(alpha);
b0 = 1/(sum(beta)+nl);
Hsu1 = a0*b0*(alpha*beta');

%%%%% construct Hsu2 %%%%%
Hsu2 = zeros(ns,nu);
for i = 1:C
    ac = 1/sum(alpha(S_Label==lbl_idx(i)));
    bc = 1/(sum(beta(U_Label==lbl_idx(i)))+sum(L_Label==lbl_idx(i)));
    alpha_c = alpha(S_Label==lbl_idx(i));
    beta_c = beta(U_Label==lbl_idx(i));
    Hsu2(S_Label==lbl_idx(i),U_Label==lbl_idx(i)) = ac*bc*(alpha_c*beta_c');
end

%%%%% construct Hsu3 %%%%%
Hsu3 = zeros(ns,nu);
for i = 1:C
    tmp1 = sum(alpha(S_Label==lbl_idx(i)));
    tmp2 = sum(beta(U_Label==lbl_idx(i)));
    nlc = sum(L_Label==lbl_idx(i));
    ec = 1/(tmp1*tmp2+tmp2*nlc+nlc*tmp1);
    alpha_c = alpha(S_Label==lbl_idx(i));
    beta_c = beta(U_Label==lbl_idx(i));
    Hsu3(S_Label==lbl_idx(i),U_Label==lbl_idx(i)) = ec*(alpha_c*beta_c');
end