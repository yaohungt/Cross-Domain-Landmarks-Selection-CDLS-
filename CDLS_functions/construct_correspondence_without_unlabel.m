% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)

function [Hs1,Hs2,Hs3,Hsl1,Hsl2,Hsl3,Hsu1,Hsu2,Hsu3] = construct_correspondence_without_unlabel(S_Label,L_Label,nu)

%%%%% Number of Categories %%%%%
lbl_idx = unique(S_Label);
C = length(lbl_idx);

%%%%% number of instances %%%%%
ns = length(S_Label);
nl = length(L_Label);

%%%%% construct Hs1 %%%%%
Hs1 = (1/(ns*ns))*ones(ns,ns);

%%%%% construct Hs2 %%%%%
Hs2 = zeros(ns,ns);
for i = 1:C
    nsc = sum(S_Label==lbl_idx(i));
    Hs2(S_Label==lbl_idx(i),S_Label==lbl_idx(i)) = (1/(nsc*nsc))*ones(nsc,nsc);
end

%%%%% construct Hs3 %%%%%
Hs3 = zeros(ns,ns);
for i = 1:C
    nsc = sum(S_Label==lbl_idx(i));
    Hs3(S_Label==lbl_idx(i),S_Label==lbl_idx(i)) = (1/nsc)*eye(nsc);
end

%%%%% construct Hsl1 %%%%%
Hsl1 = (1/(ns*(nl+nu)))*ones(ns,nl);

%%%%% construct Hsl2 %%%%%
Hsl2 = zeros(ns,nl);
for i = 1:C
    nsc = sum(S_Label==lbl_idx(i));
    nlc = sum(L_Label==lbl_idx(i));
    Hsl2(S_Label==lbl_idx(i),L_Label==lbl_idx(i)) = (1/(nsc*nlc))*ones(nsc,nlc);
end

%%%%% construct Hsl3 %%%%%
Hsl3 = zeros(ns,nl);
for i = 1:C
    nsc = sum(S_Label==lbl_idx(i));
    nlc = sum(L_Label==lbl_idx(i));
    Hsl3(S_Label==lbl_idx(i),L_Label==lbl_idx(i)) = (1/(nsc*nlc))*ones(nsc,nlc);
end

%%%%% construct Hsu1 %%%%%
Hsu1 = (1/(ns*(nl+nu)))*ones(ns,nu);

%%%%% construct Hsu2 and Hsu3 %%%%%
Hsu2 = zeros(ns,nu);
Hsu3 = zeros(ns,nu);