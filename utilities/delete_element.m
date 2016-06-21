function [X] = delete_element(X, element, dim)
% Safely deletes elementsn 'element' out of the dimension 'dim' from the matrix X.
% In case the matrix is smaller then the element to delete, nothing is
% done.
% If more than 1 dim is given, the same is repeated for all dim.

n = ndims(X);
desc = repmat({':'},1,n);
for d = 1:numel(dim)
    idx = element <= size(X,dim(d));
    desc_act = desc;
    desc_act{dim(d)} = element(idx);
    X(desc_act{:}) = [];
end

end