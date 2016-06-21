
classdef node < matlab.mixin.Copyable

properties
    % network parameters
    id_child = []; % M_child x 2 % id at (I) node array (c/g) of the layer + (II) data array (c/g) of the layer
    % id_parent = []; % M_parent x 2 % id at (I) node array (c/g) of the layer + (II) data array (c/g) of the layer
    
    % probability distribution parameters
    K = 10;% K = 1 for binary data
    % Pi not needed, alpha takes its role
    % Pi = []; % K_own x 1
    Mu = {}; % K_own x K_child for disc child, K_own x 1 for cont child; in total M_child
    Sigma = {}; % empty for disc, K_own x 1 for cont
    
    % inference
    beta = []; % p(x_in(l) | h_l) size: N x K_own
    beta_log_const = []; % size: N x 1
    alpha = [];
    sigma = []; % N x K_own: sigmas for continuous nodes; otherwise empty
    
    a = []; % p(h_parent(l) | x_out(l)) size: N x K_parent
%    a = {}; % p(h_l | x_out(child(l))) size: {N x K_own} x M_child
%     b = [];  % p(x_in(c1) | h_l) size: N x K_own
%     b_log_const = []; % b normalization constant; size: N x 1
%     % b_orig == b * exp(b_log_const)

    prior = [] % only for class balance
end

methods
    % constructor
    function obj = node(id_child, K)               
        obj.id_child = id_child;
        assert(~mod(K,1)); % check for integer
        obj.K = K;
    end
    function set_beta(obj, beta, beta_log_const)
        % check that dimensionality is correct
        assert(obj.K == size(beta,2));
        if obj.K > 1
            % normalize beta
            [beta, s] = normalize_convex(beta, 2);
            obj.beta_log_const = beta_log_const + log(s);
        else
            % for continuous data there is no beta_log_const
            assert(isempty(beta_log_const));
            obj.beta_log_const = [];
        end
        obj.beta = beta;
    end
    function set_alpha(obj, alpha, sigma)
        % check that dimensionality is correct
        assert(obj.K == size(alpha,2));
        is_nan = all(isnan(alpha),2);
        if obj.K > 1
            % normalize alpha
            s = sum(alpha,2);
            sinv = 1./s;
            assert(all(isfinite(sinv) | is_nan)); % sanity check
            alpha = bsxfun(@times, alpha, sinv);           
        end
        obj.sigma = sigma;
        obj.alpha = alpha;
    end
    function set_a(obj, a)    
        if obj.K > 1
            % normalize alpha
            is_nan = all(isnan(a),2);
            s = sum(a,2);
            sinv = 1./s;
            assert(all(isfinite(sinv) | is_nan)); % sanity check
            a = bsxfun(@times, a, sinv);
        end
        obj.a = a;
    end
    function obj = clear_data(obj, clear_alpha)
        obj.beta = [];
        obj.beta_log_const = [];
        if clear_alpha
            % clear alpha only for non-roots
            obj.alpha = [];
        end
        obj.a = [];
    end
    function [beta, beta_log_const] = get_beta(obj)
        beta = obj.beta;
        beta_log_const = obj.beta_log_const;
        if isempty(beta_log_const)
            N = size(beta,1);
            beta_log_const = zeros(N,1);
        end
    end
    function [alpha, sigma] = get_alpha(obj)
        alpha = obj.alpha;
        assert(~isempty(alpha));
        sigma = obj.sigma;
    end
    function a = get_a(obj)
        a = obj.a;
        assert(~isempty(a));
    end
    function set_data(obj, X)
        
        if iscell(X)
            assert(numel(X)==1);
            distrib = X{1};
            [N, M] = size(distrib);
            if (obj.K > 1) && (obj.K == M)
                assert(all( abs(sum(distrib,2)-1) < 1e-12 ));
                obj.set_beta(distrib, zeros(N,1));
                return;
            else
                X = X{1};
            end
        end
        
        [N, M] = size(X);
        assert(1 == M);
        assert(isempty(obj.id_child)); % only set data for leaves
        if obj.K == 1
            % continuous data
            obj.set_beta(X, []);
        else
            % discrete data
            assert(all(mod(X,1)==0 | isnan(X))); % check for integer
            assert(all((X > 0 & X <= obj.K) | isnan(X)));
            distrib = zeros(N, obj.K);
            ind = sub2ind([N obj.K], (1:N)', X);
            ind = ind(~isnan(ind)); % ignore NaN's
            distrib(ind) = 1;
            
            %distrib_nan = bsxfun(@times, isnan(X), ones(1,obj.K));
            % distrib = distrib + distrib_nan;
            distrib(isnan(X),:) = nan;
            obj.set_beta(distrib, zeros(N,1));
        end
    end
    function add_child(obj, id_child, beta_child)
        obj.id_child = cat(1, obj.id_child, id_child);
        m_child = numel(obj.id_child);
        % initialize the child distribution according to the current
        % distribution of the parent (q) and beta_child:
        q = calc_q(obj);
        [Mu_new, Sigma_new] = hpmm2.calc_MuSigma(q, beta_child);
        obj.Mu{m_child} = Mu_new;
        obj.Sigma{m_child} = Sigma_new;
    end
    function id_child = children(obj)
        id_child = obj.id_child;
    end
    function obj = delete_node_id(obj, id_node)
        % we just need to reindex all children ids >= node_id
        idx = obj.id_child >= (id_node+1);
        obj.id_child(idx) = obj.id_child(idx)-1;
    end
    function K = get_K(obj)
        K = obj.K;
    end
    function init_distribution(obj, beta_children, K)
        if exist('K','var') && ~isempty(K)
            obj.K = K;
        end
        
        M = size(obj.id_child,1);
        if M > 0           
            % init for node with children
            assert(M == numel(beta_children));
            N = size(beta_children{1},1);
            % select randomly K samples out of N as means
            idx_mu = randperm(N,obj.K);
            for c = 1:M
                beta_act = beta_children{c};
                
                obj.Mu{c} = beta_act(idx_mu,:);                
                K_child = size(beta_act,2);
                if K_child == 1
                    % only for continuous children
                    
                    if M == 1
%                         % kmeans strategy
%                         idx_clust = kmeans(beta_act,obj.K);

%                         % equidist strategy
%                         beta_act = sort(beta_act);
%                         idx_clust = equidistant_subsets( N, obj.K);
%                         mu = zeros(obj.K,1);
%                         sigma = zeros(obj.K,1);
%                         for k = 1:obj.K
%                             beta_act_k = beta_act(idx_clust == k,1);
%                             mu(k) = mean(beta_act_k,1);
%                             sigma(k) = var(beta_act_k,0,1);
%                             if sigma(k) == 0
%                                 sigma(k) = var(beta_act,0,1);
%                             end
%                         end
                        start = min(beta_act,[],1);
                        range = max(beta_act,[],1) - start;
                        step = range / (obj.K+1);
                        mu = ((1:obj.K)' .* step) + start;
                        sigma = ones(obj.K,1) .* (nanvar(beta_act,0,1) / obj.K);
                        obj.Mu{c} = mu;
                        obj.Sigma{c} = sigma;
                    else                                        
                        X_var = nanvar(beta_act,0,1);
                        obj.Sigma{c} = repmat(X_var, obj.K, 1);
                        
                        % check for any NaNs in Mu
                        mu_act = obj.Mu{c};
                        assert(numel(mu_act) == obj.K);
                        idx_nan = isnan(mu_act);
                        if any(idx_nan)
                            beta_act(idx_mu,:) = []; % firs remove the previously selected mu
                            beta_act = beta_act(~isnan(beta_act),1); % then remove NaNs
                            N = size(beta_act,1);
                            idx_mu_new = randperm(N,sum(idx_nan));
                            mu_act(idx_nan) = beta_act(idx_mu_new,1);
                            obj.Mu{c} = mu_act;
                        end
                    end
                else
                    obj.Sigma{c} = [];
                    obj.Mu{c} = normalize_convex(0.4 + 0.2*rand(obj.K,K_child),2);
                end
                assert(~any(isnan(obj.Mu{c}(:))));
                assert(~any(isnan(obj.Sigma{c}(:))));
            end
            % uniform priors
            obj.alpha = ones(1,obj.K)/obj.K;
        else
            % init for discrete data node
            %(continuous data node needs a parent!)
            assert(obj.K > 1);
            beta = get_beta(obj);
            % beta: N x K_own
            beta = normalize_convex(beta,2);
            pi_new = nansum(beta,1);
            pi_new = normalize_convex(pi_new,2);
            obj.prior = pi_new;
            assert(~any(isnan(pi_new(:))));
            
            % check for 0 states and replace them with small values
            pi_new = hpmm2.replace_zeros(pi_new);
            obj.alpha = pi_new;
        end
    end
    function [log_b] = calc_log_b(obj, beta_child, m)
        [log_b] = hpmm2.calc_log_b(beta_child, obj.Mu{m}, obj.Sigma{m});
    end
    function [alpha_child, a_child, sigma_child] = calc_alpha_child(obj, alpha_log, log_b_siblings, m_child)
        % alpha: N x K_own or 1 x K_own (depending if root or not)
        % log_b_siblings: N x K_own
        K_own = size(obj.Mu{m_child},1);
        assert(K_own == size(alpha_log,2));
        assert(K_own == size(log_b_siblings,2));
        
        % discrete child: multiply alpha*b_siblings with CPT
        % continuous child: get expected value over the means (same
        % operation)
        
        % use bsxfun since alpha can be N x K_own or 1 x K_own
        a_child_log = bsxfun(@plus, alpha_log, log_b_siblings);
        a_child = normalize_convex_log(a_child_log,2);
        
        [alpha_child, sigma_child] = hpmm2.calc_alpha(a_child, obj.Mu{m_child}, obj.Sigma{m_child});
    end
    function update_cpd(obj, beta_child, a_child, m_child)
        % beta_child: N x K_child
        % a_child: N x K_own
        % obj.Mu{m_child}: K_own x K_child
        
        K_child = size(obj.Mu{m_child},2);
        assert(K_child == size(beta_child,2));

        if K_child > 1
            % for discrete child
            
            % remove nans
            nan_beta = all(isnan(beta_child),2);
            nan_a = all(isnan(a_child),2);
            nan_any = nan_beta | nan_a;
            beta_child(nan_any,:) = [];
            a_child(nan_any,:) = [];
            
            % outer product:
            abeta = bsxfun(@times, permute(beta_child,[3 2 1]), permute(a_child,[2 3 1]));
            % abeta: K_own x K_child x N
            % pairwise product:
            q = bsxfun(@times, abeta, obj.Mu{m_child});
            % normalize regarding joint (K_own and K_child)
            q = normalize_convex(q, [1 2]);
            % q = q(h_own, h_child)
            
            Mu_new = sum(q,3);
            
            % check for empty states
            s = sum(Mu_new,2);
            if any(s == 0)                
                % warning('Empty state detected. Reinitialize');
                idx = s == 0;
                Mu_new(idx,:) = normalize_convex(0.4 + 0.2*rand(sum(idx),K_child),2);
            end
            
            % normalize regarding K_child
            Mu_new = normalize_convex(Mu_new, 2);
            
            % check for 0 states and replace them with small values
            Mu_new = hpmm2.replace_zeros(Mu_new);
            
            obj.Mu{m_child} = Mu_new;
        else
            % for continuous child
            q = calc_q(obj);
            
            % remove NaNs
            nan_beta = all(isnan(beta_child),2);
            nan_q = all(isnan(q),2);
            nan_any = nan_beta | nan_q;
            beta_child(nan_any,:) = [];
            q(nan_any,:) = [];
            
            % get the normalization constant regarding N
            N_k = sum(q,1)';
            % check for zeros
            N_k(N_k == 0) = 1;
            % N_k: 1 x K_own
            % q: N x K_own
            % beta_child: N x K_child(=1)
            % N_kinv: K_own x 1
            Mu_new = (q' * beta_child) ./ N_k;
            % Mu_new: K_own x 1
            dXsq = bsxfun(@minus, Mu_new', beta_child).^2;
            % dXsq: N x K_own
            Sigma_new = sum(q' .* dXsq', 2) ./ N_k;
            %assert(all(Sigma_new > 0));
            % keep small Sigmas fixed
%             idx_zero = sqrt(Sigma_new) <= eps;
%             Sigma_new(idx_zero) = obj.Sigma{m_child}(idx_zero);
            
            idx_zero = Sigma_new < eps;
            Sigma_new(idx_zero) = eps;
            
            obj.Mu{m_child} = Mu_new;
            obj.Sigma{m_child} = Sigma_new;
        end
        assert(~any(isnan(Mu_new(:))));
    end
    function lklhd_all = update_prior(obj)
        % Only makes sense for root node!
        % (otherwise the alpha is overwritten during the alpha_pass)
        % as a by-product we return the log-likelihood for the tree
        [q, lklhd_old] = calc_q(obj);
        pi_new = nansum(q, 1);
        pi_new = normalize_convex(pi_new, 2);
        
        % check for 0 states and replace them with small values
        pi_new = hpmm2.replace_zeros(pi_new);
        
        obj.alpha = pi_new;
        assert(~any(isnan(pi_new(:))));
        % get likelihood for the updated alpha
        [~, lklhd] = calc_q(obj);
        lklhd_all = nanmean(lklhd,1);
        lklhd_all_old = nanmean(lklhd_old,1);
        assert(lklhd_all_old - lklhd_all < 1e-10); % EM guarantees improvement
    end
    function [q, lklhd] = calc_q(obj)
        % Calculates the marginal posterior q for this node l, as well as
        % the conditional log-likelihood 
        % lklhd = ln p(x(n)_in(l) | x(n)_out(l)): N x 1
        % q: N x K_own
        % Prerequisites: alpha and beta must be set
        % alpha: 1 x K_own OR N x K_own
        % beta: N x K_own
        
        if ~isempty(obj.beta)
            % regular case
            q = bsxfun(@times, obj.alpha, obj.beta);
            [q, c] = normalize_convex(q, 2);
            % lklhd = log sum_k( alpha(k)*beta(k) )
            lklhd = log(c);
            if ~isempty(obj.beta_log_const)
                % in case that beta_orig = beta * exp(beta_log_const), we need to
                % add it to the log
                lklhd = lklhd + obj.beta_log_const;
            end
        else
            % if no evidence has been passed into this tree
            q = [];
            lklhd = 0;
        end
    end
    function plot(obj, beta_child, m_child)
        [N, K_child] = size(beta_child);
        assert(K_child == 1); % this works only for continuous data
        
        %hist(beta_child,100);
        b_max = max(beta_child);
        b_min = min(beta_child);
        num_bin = 100;
        step = (b_max - b_min)/num_bin;
        binranges = b_min:step:b_max;
        bincounts = histc(beta_child,binranges);
        
        %clf();
        cla();
        bar(binranges,bincounts,'histc');
        y_scale_count = max(bincounts);
        
        hold on;
        K_own = obj.K;
        S = obj.Sigma{m_child};
        y_scale_sigma = max(S(:,1));
        for k = 1:K_own
            mu = [obj.Mu{m_child}(k,1) 0];            
            covariance = repmat(S(k,1),1,2) .* [1 (y_scale_count/y_scale_sigma)];
            PComponent = obj.alpha(k);
            plot_mean_covariance(covariance, mu, PComponent);
        end
    end
    function obj = normalize_beta(obj)
        K_own = size(obj.beta,2);
        if K_own > 1
            obj.beta = normalize_convex(obj.beta, 2);
        end
        obj.beta_log_const = [];
    end
    obj = update_data(obj);
    obj = optimize_param(obj);
end % methods
end % classdef
