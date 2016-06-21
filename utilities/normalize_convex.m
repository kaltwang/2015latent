function [ x, s ] = normalize_convex( x, dim )
% normalizes x to sum to 1 along dimensions(s) dim
% asserts that x does NOT sums to 0
% also returns the sum s

s = sum(x, dim(1));
for d = dim(2:end)
    s = sum(s, d);
end
assert(all(s(:) ~= 0));
x = bsxfun(@rdivide, x, s);
end

