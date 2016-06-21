function [N, M] = data_get_NM(X)
M = size(X,2);
if ~iscell(X)
    N = size(X,1);
else
    N = size(X{1},1);
end
end