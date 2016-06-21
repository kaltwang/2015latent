function [ alpha_child, sigma_child ] = calc_alpha(a_child, Mu, Sigma)
% a_child: N x K_own
% Mu: K_own x K_child
assert(~any(isnan(Mu(:))));

K_child = size(Mu,2);

alpha_child = a_child * Mu;
% alpha_child: N x K_child

if K_child > 1
    % normalize (only for discrete child)
    % alpha_child = normalize_convex(alpha_child, 2);
    % its a convex sum over convex vectors; the result must be
    % convex
    assert(all( (abs(sum(alpha_child,2)-1) < 1e-12) | all(isnan(alpha_child),2) ));
    sigma_child = [];
else
    % variance within-components
    sigma_child_wc = a_child * Sigma;
    % variance between-components
    sigma_child_bc = sum(a_child .* bsxfun(@minus, alpha_child, Mu').^2,2);
    sigma_child = sigma_child_wc + sigma_child_bc;
end

end