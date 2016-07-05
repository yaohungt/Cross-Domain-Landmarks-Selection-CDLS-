% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)
%
% Main code of Cross-Domain Landmarks Selection (CDLS)

function [acc] = CDLS(Data,param)

%%%%% Input from Data & param %%%%%
T = Data.T;
Ttest = Data.Ttest;
S = Data.S;
T_Label = Data.T_Label;
Ttest_Label = Data.Ttest_Label;
S_Label = Data.S_Label;

iter = param.iter;
scale = param.scale;
delta = param.delta;
PCA_dimension = param.PCA_dimension;

%%%%% Initialization %%%%%
accVec = zeros(iter,1);
U_Label = [];
alpha = ones(length(S_Label),1);
beta  = ones(length(Ttest_Label),1);

%%%%% Construct Pt by PCA %%%%%
[Pt,~,~] = princomp([T';Ttest']);
Pt = Pt(:,1:PCA_dimension);

%%%%% Project Target Intances on Target PCA Subspace %%%%%
co_Xt = Pt'*T;
co_Xtest = Pt'*Ttest;

%%%%% Iteration Start %%%%%
for i = 1:iter

	%%%%% Construct Centering Matrix %%%%%
    if ~isempty(U_Label)
        [Hs1,Hs2,Hs3,Hsl1,Hsl2,Hsl3,Hsu1,Hsu2,Hsu3] = construct_correspondence_with_unlabel(S_Label,T_Label,U_Label,alpha,beta);
    else
        [Hs1,Hs2,Hs3,Hsl1,Hsl2,Hsl3,Hsu1,Hsu2,Hsu3] = construct_correspondence_without_unlabel(S_Label,T_Label,length(Ttest_Label));
    end
	
	%%%%% Construct Ps %%%%%    
    inverse_part = S*(Hs1+Hs2+Hs3)*S';
    non_inverse_part = S*(Hsl1+Hsl2+Hsl3)*T' + S*(Hsu1+Hsu2+Hsu3)*Ttest';
    
    regularizer_weight = 0.5;
	Ps = (( inverse_part + regularizer_weight*eye(size(S,1)) )\(non_inverse_part))*Pt;
	
	%%%%% Project Source Intances on Target PCA Subspace %%%%%
	co_Xs = Ps'*S;
    
    %%%%% SVM Test %%%%%
    [accVec(i),U_Label] = SVM_test(scale,alpha,co_Xs',S_Label,co_Xt',T_Label,co_Xtest',Ttest_Label);
    
    fprintf('Accuracy of iteration %d is %f \n', i, accVec(i));
    
    %%%%% Calculate alpha and beta %%%%%
    if delta ~= 1
        [alpha,beta] = QP_choose_source_unlabel(co_Xs,co_Xt,co_Xtest,S_Label,T_Label,U_Label,delta);
    end 
    
end

acc = accVec(end);

