function [log_b] = calc_log_b(beta_child, Mu, Sigma)
    assert(~any(isnan(Mu(:))));
    % beta_child: N x K_child
    K_child = size(beta_child,2);
    N = size(beta_child,1);
    if K_child > 1
        % discrete child: multiply beta_child with CPT
        % Mu: K_own x K_child
        log_b = log(beta_child * Mu');
    else
        % continuous child: Gaussian density
        assert(1 == size(beta_child,2));

        Sigma = Sigma';
        % Sigma: 1 x K_own
        L = sqrt(Sigma); % a vector
        assert(all(L > eps));
        logDetSigma = log(Sigma);
        Xcentered = bsxfun(@minus, beta_child, Mu');
        % Xcentered: N x K_own
        xRinv = bsxfun(@times,Xcentered , (1./ L));
        mahalaD = xRinv.^2;
        log_lh = bsxfun(@plus, -0.5 * mahalaD, ...
        (-0.5 *logDetSigma) - log(2*pi)/2);
        % log_lh: N x K_own
        log_b = log_lh;
    end
    
    % handle NaNs
    idx_nan = any(isnan(beta_child),2);
    K_own = size(Mu,1);
    % log_b(idx_nan,:) = log(1/K_own);
    log_b(idx_nan,:) = 0; 
end