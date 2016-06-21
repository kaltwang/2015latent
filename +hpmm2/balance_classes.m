function [alpha_new] = balance_classes(alpha, prior)
K = size(alpha,2);
if K==1
   % continuous case, do nothing
   assert(isempty(prior));
   alpha_new = alpha;
   return;
end

prior_inv = ones(1,K) - prior;
alpha_new = bsxfun(@times, alpha, prior_inv);
alpha_new = normalize_convex(alpha_new,2);
end