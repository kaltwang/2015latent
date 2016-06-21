function [X] = replace_zeros(X)
% check for 0 states and replace them with small values
    idx_zero = X(:) == 0;
    if any(idx_zero)
        val_zero = min(min(X(~idx_zero)), eps);
        X(idx_zero) = val_zero;
        X = normalize_convex(X, 2);
    end
end