function [X, idx_nan] = set_nans_zero(X)

idx_nan = any(isnan(X),2);
X(idx_nan,:) = 0;

end