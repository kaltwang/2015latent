function [Mu, Sigma] = calc_MuSigma(q_parent, beta_child)
% q_parent: N x K_parent
% beta_child: N x K_child

K_child = size(beta_child,2);

% remove NaNs
idx_unobserved = any(isnan(beta_child),2) | any(isnan(q_parent),2);
beta_child(idx_unobserved,:) = [];
q_parent(idx_unobserved,:) = []; 

% get the normalization constant regarding N
N_p = sum(q_parent,1)';
% check for zeros
idx_zero = N_p <= 1e-9;
N_p(idx_zero) = 1;
% N_p: K_parent x 1

if K_child > 1
    % normalize beta_child
    beta_child = normalize_convex(beta_child,2);
end

Mu = bsxfun(@rdivide,q_parent' * beta_child, N_p);
% Mu: K_parent x K_child
if K_child == 1
    dXsq = bsxfun(@minus, Mu', beta_child).^2;
    % dXsq: N x K_parent
    Sigma = sum(q_parent' .* dXsq', 2) ./ N_p;
    % L = sqrt(Sigma); % a vector
    % assert(all(L > eps));
    
%     Sigma_all = wmean(Sigma, 1, N_p);   
%     idx_zero = sqrt(Sigma) <= eps;    
%     Sigma(idx_zero) = Sigma_all;
    
    idx_zero = Sigma < eps;    
    Sigma(idx_zero) = eps;
else
    Sigma = [];
    num_zero = sum(idx_zero);
    if num_zero > 0
        % initialize empty states randomly
        Mu(idx_zero,:) = normalize_convex(0.4 + 0.2*rand(num_zero,K_child),2);
    end
    
    % check for 0 states and replace them with small values
    Mu = hpmm2.replace_zeros(Mu);
    
    assert(all(abs(sum(Mu,2) - 1) < 1e-5));
end

assert(~any(isnan(Mu(:))));

end