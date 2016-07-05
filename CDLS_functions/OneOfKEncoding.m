function Yk = OneOfKEncoding(Y)
% class labels in Y are 1,2,3,...

Yk = full(sparse(1:length(Y), Y, ones(length(Y),1)));
