function [ x, s_log ] = normalize_convex_log( x_log, dim )
% normalizes exp(x_log) to sum to 1 along dimensions(s) dim
% asserts that x does NOT sums to 0
% also return the sum in log_space s_log

m = max(x_log, [], dim(1));
for d = dim(2:end)
    m = max(m, [], d);
end
x_log = bsxfun(@minus, x_log, m);
x = exp(x_log);
[x, s] = normalize_convex( x, dim );
s_log = m + log(s);
end

