function [X, idx_nan] = remove_nans(X)

idx_nan = any(isnan(X),2);
X(idx_nan,:) = [];

end