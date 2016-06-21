
classdef graph < matlab.mixin.Copyable

properties
    % Parameters
    nodes = hpmm2.node.empty(); % L x 1
    N = 0; % number of actual data points
    K_default = 10;
    M = 0; % number of observed inputs
    
    % struct learning options
    find_best_K = false;
    recursive_EM_after_struct_change = true;
    recursuve_EM_after_new_closed = false;
    recursuve_EM_final = false;
    lklhd_cond_mindiff = -0.01; % minimum differences in the conditional likelihood
    lklhd_cond_single = false;
    strategy = 'always combine';
    minscore = 0.01;
    focus_dim = [];
    no_new_hidden_nodes = false;
    close_last_first = false;
    lower_layers_first = false;
    
    ind_u = [];
    combine_observed_first = false;
    
    % EM options
    mindiff = 1e-6;
    maxcount = 500;
    dispevery = 20;
    EM_restarts = 5;
    
    % prediction options
    get_data_mode = 'expected value';
    balance_classes = false;
    
    % structure learning variables
    nodes_top = []; % L x 1; binary: nodes is either on top or not
    nodes_parent = []; % L x 1; either contains itself as possbile parent or the parent in case of continuous nodes
    nodes_lklhd = []; % L x 1
    nodes_lklhd_diff = []; % L x L
    nodes_inf_dist = [];
    nodes_inf_dist_cramer = [];
    nodes_corr = [];
    nodes_tree_id = []; % root node id of the tree
    nodes_failed = []; % L x L x 3; remember which nodes we already tried to join (the 3rd dim stands for the join cases 1-3)
    nodes_closed = []; % L x 1
    
    nodes_focus = [];
    nodes_layer = [];
    node_changed_last = [];
    
    % history
    history_keep = false;
    history = [];
    
    plot_EM = false;
    plot_graph = true;
    plot_lklhd_diff = true;
    
    % validation and training copies
    X_train = [];
    X_validation = [];
    
end % properties

methods
    % constructor
    function obj = graph(varargin)               
        obj = set(obj,varargin{:});
    end
    
    obj = set( obj, varargin );
    
    plot( obj, mapping  ); % plot the graph
    function id_node = add_node( obj, id_child, K )
        obj.nodes(end+1,1) = hpmm2.node(id_child, K);
        id_node = size(obj.nodes,1);
        
%         obj.nodes_closed(id_node,1) = false;
%         obj.nodes_focus(id_node,1) = any(obj.nodes_focus(id_child));
%         if ~isempty(id_child)
%             obj.nodes_layer(id_node,1) = max(obj.nodes_layer(id_child));
%         else
%             obj.nodes_layer(id_node,1) = 1;
%         end
    end
    function [lklhd_best] = node_find_K( obj, id_parent)
        if obj.find_best_K
            id_child = obj.children(id_parent);
            P_child = 0;
            for c = 1:length(id_child)
                P_child = P_child + get_num_parameters(obj, id_child(c), true);
            end
            K = 2;
            next = true;
            BIC_best = -Inf;
            lklhd_best = -Inf;
            node_best = [];
            K_best = 2;

            obj.normalize_beta(id_parent);

            while next
                init_node_distribution( obj, id_parent, K );
                P_prior = K-1;
                P_parent = get_num_parameters(obj, id_parent, false);
                P = P_parent + P_prior;
                lklhd = EM(obj, id_parent, false);
                BIC = obj.N * lklhd - 0.5 * P * (log(obj.N) - log(2*pi));

                if BIC > BIC_best
                    node_best = copy(obj.nodes(id_parent));
                    BIC_best = BIC;
                    lklhd_best = lklhd;
                    K_best = K;
                    K = K+1;
                else
                    next = false;
                end
            end

            % recover best node
            delete(obj.nodes(id_parent));
            obj.nodes(id_parent) = node_best;
            % recover child values
            beta_pass( obj, id_parent, false );
            alpha_pass( obj, id_parent, false );
            lklhd_best = calc_lklhd(obj, id_parent);
            disp(['Best K = ' num2str(K_best) ' for node ' num2str(id_parent) ' with BIC = ' num2str(BIC_best)]);
        else
            init_node_distribution( obj, id_parent, obj.K_default );
            lklhd_best = EM(obj, id_parent, false);
        end
    end
    function delete_node(obj, id_node)
        % first check that id_node has no parent
        p = parents( obj );
        assert(p(id_node) == 0);
        % delete the node handle
        delete(obj.nodes(id_node));
        % clear the handle variable
        obj.nodes(id_node) = [];
        
        % decrease all ids >= id_node+1 by1       
        L = size(obj.nodes,1);
        for l = 1:L
            obj.nodes(l).delete_node_id( id_node );
        end
        
        % update node properties
%         if size(obj.nodes_top,1) >= id_node
%             obj.nodes_top(id_node,:) = []; 
%         end
%         if size(obj.nodes_parent,1) >= id_node
%             obj.nodes_parent(id_node,:) = [];
%         end
%         if size(obj.nodes_lklhd,1) >= id_node
%             obj.nodes_lklhd(id_node,:) = [];
%         end
%         if size(obj.nodes_lklhd_diff,1) >= id_node
%             obj.nodes_lklhd_diff(id_node,:) = [];
%             obj.nodes_inf_dist(id_node,:) = [];
%             obj.nodes_inf_dist_cramer(id_node,:) = [];
%         end
%         if size(obj.nodes_lklhd_diff,2) >= id_node
%             obj.nodes_lklhd_diff(:,id_node) = [];
%             obj.nodes_inf_dist(:,id_node) = [];
%             obj.nodes_inf_dist_cramer(:,id_node) = [];
%         end
%         if size(obj.nodes_tree_id,1) >= id_node
%             obj.nodes_tree_id(id_node,:) = [];
%         end
        
        obj.nodes_top     = delete_element(obj.nodes_top,     id_node, 1);
        obj.nodes_parent  = delete_element(obj.nodes_parent,  id_node, 1);
        obj.nodes_lklhd   = delete_element(obj.nodes_lklhd,   id_node, 1);
        obj.nodes_tree_id = delete_element(obj.nodes_tree_id, id_node, 1);
        obj.nodes_closed  = delete_element(obj.nodes_closed,  id_node, 1);
        obj.nodes_focus   = delete_element(obj.nodes_focus,   id_node, 1);
        obj.nodes_layer   = delete_element(obj.nodes_layer,   id_node, 1);
        
        obj.nodes_lklhd_diff      = delete_element(obj.nodes_lklhd_diff,      id_node, [1 2]);
        obj.nodes_inf_dist        = delete_element(obj.nodes_inf_dist,        id_node, [1 2]);
        obj.nodes_inf_dist_cramer = delete_element(obj.nodes_inf_dist_cramer, id_node, [1 2]);
        obj.nodes_corr            = delete_element(obj.nodes_corr,            id_node, [1 2]);
        obj.nodes_failed          = delete_element(obj.nodes_failed,          id_node, [1 2]);
        
        % reindex nodes_parent
        idx = obj.nodes_parent >= id_node;
        obj.nodes_parent(idx) = obj.nodes_parent(idx) - 1;
        
        % reindex nodes_tree_id
        idx = obj.nodes_tree_id >= id_node;
        obj.nodes_tree_id(idx) = obj.nodes_tree_id(idx) - 1;
               
    end
    function add_child(obj, id_parent, id_child )
        % add child and init its distribution
        beta_child = obj.nodes(id_child).get_beta();
        obj.nodes(id_parent).add_child(id_child, beta_child);
        
        obj.nodes_focus(id_parent,1) = any(obj.nodes_focus([id_parent; id_child]));
    end
    function ids = ids( obj )
        ids = (1:size(obj.nodes,1))';
    end
    function p = parents( obj )
        L = size(obj.nodes,1);
        p = zeros(1,L);
        for l = 1:L
            id_child = obj.nodes(l).children();
            if ~isempty(id_child)
                p(id_child) = l;
            end
        end
    end
    function id_child = children( obj, id_parent )
        id_child = children(obj.nodes(id_parent));
    end
    function K = get_K( obj, id_node )
        if ~exist('id_node','var') || isempty(id_node)
            % use all nodes
            id_node = 1:size(obj.nodes,1);            
        end
        L = numel(id_node);
        K = zeros(L,1);
        for l = 1:L
            K(l) = obj.nodes(id_node(l)).get_K();
        end
    end
    function A = get_A(obj)
        L = size(obj.nodes,1);
        A = eye(L);
        for l = 1:L
            id_child = obj.nodes(l).children();
            c = size(id_child,1);
            if c > 0
                ind = sub2ind([L L], repmat(l,c,1), id_child);
                A(ind) = 1;
            end
        end
    end
    function P = get_num_parameters_total( obj )
        p = parents(obj);
        roots = find(p == 0);
        P = 0;
        for r = 1:length(roots)
            % add prior parameter:
            P = P + get_K( obj, roots(r) );
            P = P + get_num_parameters( obj, roots(r), true );
        end
    end
    function P = get_num_parameters( obj, id_root, recursive )
        id_child = children( obj, id_root );
        K_own = get_K(obj, id_root);
        P = 0;
        for c = 1:length(id_child)
            K_child = get_K(obj, id_child(c));
            if K_child > 1
                % discrete child
                % parameters are the CPT with K_own * (K_child-1)
                % parameters
                P = P + K_own * (K_child-1);
            else
                % continuous child
                % parameters are the mean and the variance for each K_own
                P = P + K_own * 2;
            end
            
            if recursive
                P = P + get_num_parameters(obj, id_child(c), recursive);
            end
        end
    end
    function id_node = init_data( obj, X, K)
        [obj.N, obj.M] = hpmm2.data_get_NM(X);
        assert(isequal([obj.M 1], size(K)));
        obj.nodes.delete();
        obj.nodes = hpmm2.node.empty();
        
        id_node = zeros(obj.M,1);
        for m = 1:obj.M
            id_node(m) = obj.add_node([], K(m) );            
        end
        assert(isequal(id_node, (1:obj.M)'));
        
        set_data( obj, X, id_node );
    end
    function obj = set_data( obj, X, id_node)
        [obj.N, M] = hpmm2.data_get_NM(X);
        for m = 1:M          
            set_data( obj.nodes(id_node(m)), X(:,m) );
        end
    end
    function obj = clear_data( obj )
        L = size(obj.nodes,1);
        p = obj.parents();
        for l = 1:L
            if p(l) ~= 0
                % clear alpha only for non-roots
                clear_alpha = true;
            else
                clear_alpha = false;
            end
            obj.nodes(l) = clear_data(obj.nodes(l), clear_alpha);
        end
        obj.N = 0;
    end
    function [X, Var] = get_data( obj, id_node )        
        
        M = numel(id_node);
        X = zeros(obj.N,M);
        Var = NaN(obj.N,M);
        for m = 1:M
            [alpha, sigma] = get_alpha(obj.nodes(id_node(m)));
            if obj.balance_classes
               alpha = hpmm2.balance_classes(alpha, obj.nodes(id_node(m)).prior); 
            end
            [X(:,m), Var(:,m)] = hpmm2.distrib_to_value(alpha, sigma, obj.get_data_mode);
        end
    end
    function init_node_distribution( obj, id_node, K)
        if ~exist('K','var') || isempty(K)
            K = [];
        end
        id_child = obj.nodes(id_node).children();
        M = size(id_child,1);
        beta_children = cell(M,1);
        for m = 1:M
            beta_act = obj.nodes(id_child(m)).get_beta();
            if isempty(beta_act)
                error(['No beta for child ' num2str(id_child(m)) ' of node ' num2str(id_node) '. Beta pass needed first.']);
            end
            beta_children{m} = beta_act;
        end
        obj.nodes(id_node).init_distribution(beta_children, K);
    end
    function beta_pass( obj, id_root, recursive )
        id_child = obj.nodes(id_root).children();
        K = obj.nodes(id_root).get_K();
        M = size(id_child,1);
        if M > 0
            % otheriwse node is leaf, i.e. either is constant and has beta
            % already set from data or there is no beta, bc the node is
            % unobserved
            beta_log = zeros(obj.N,K);
            beta_log_const = zeros(obj.N,1);
            for m = 1:M
                if recursive
                    % recursively calculate beta of the children
                    obj.beta_pass(id_child(m), recursive);
                end
                
                [beta_child, beta_child_log_const] = obj.nodes(id_child(m)).get_beta();
                if ~isempty(beta_child)
                    % otherwise there is no data evidence within this child
                    N_act = size(beta_child,1);
                    assert(obj.N == N_act);
                    log_b = obj.nodes(id_root).calc_log_b(beta_child, m);
                    % maybe multiplying in log-domain? Do it in case of all
                    % beta getting towards zero
                    beta_log = beta_log + log_b;
                    beta_child_log_const = hpmm2.set_nans_zero(beta_child_log_const); % this is a hack! these values should be 0 instead of nan
                    beta_log_const = beta_log_const + beta_child_log_const;
                end
            end
            max_log_beta = max(beta_log,[],2);
            beta = exp(bsxfun(@minus, beta_log, max_log_beta));
            beta_log_const = beta_log_const + max_log_beta;
            
            obj.nodes(id_root).set_beta(beta, beta_log_const);
        else
            beta = obj.nodes(id_root).get_beta();
            if ~isempty(beta)
                % check that the observed data has the right N
                assert(obj.N == size(beta,1));
            end
        end
    end
    function alpha_pass( obj, id_root, recursive )
        % this function writes the alpha values to all children of id_root
        % and assumes that the aplha of id_root has already been calculated
        id_child = obj.nodes(id_root).children();
        K = obj.nodes(id_root).get_K();
        M = size(id_child,1);
        if M > 0
            % if node is a leaf, then there is nothing to do
            alpha_log = log(obj.nodes(id_root).get_alpha());
            
            % alternative way of calculating b_log_siblings runs into
            % numerical difficulties if one of the beta_child_act states
            % probabilities is zero! (Solution: move all beta variables
            % into log space, not only the normalizing constant)
%             [beta_own] = obj.nodes(id_root).get_beta();
%             beta_own_log = zeros(obj.N,K);
%             if ~isempty(beta_own)
%                 beta_own_log = log(beta_own);
%             end
            
            % alpha: 1 x K or N x K (depending if node is root or not)
            
            % pre-calculate log_b
            log_b_pre = zeros([obj.N K M]);
            for ms = 1:M
                beta_child = obj.nodes(id_child(ms)).get_beta();
                if ~isempty(beta_child)
                    log_b_pre(:,:,ms) = obj.nodes(id_root).calc_log_b(beta_child, ms);
                    % log_b_pre(:,:,ms) = hpmm2.set_nans_zero(log_b_pre(:,:,ms));
                end
            end
            
            for m = 1:M
                % for each child:
                % (1) multiply b of all other children
                % (2) multiply with alpha
                % (3) matrix product with CPT OR Gaussian means (to calculate
                % the Expected value of the mixture)
                % b_log_siblings = zeros(obj.N,K);
                b_log_siblings_pre = zeros(obj.N,K);
                
                for ms = 1:M
                    if ms ~= m
                        % check if is sibling
%                         beta_child = obj.nodes(id_child(ms)).get_beta();
%                         if ~isempty(beta_child)
%                             % otherwise there is no evidence fromm this child
%                             b_log_siblings = b_log_siblings + obj.nodes(id_root).calc_log_b(beta_child, ms);
%                         end
                        b_log_siblings_pre = b_log_siblings_pre + log_b_pre(:,:,ms);
                    end
                end
%                 diff = abs(normalize_convex_log( b_log_siblings, 2) - normalize_convex_log( b_log_siblings2, 2));
%                 diff_max = max(diff(:));
%                 assert(diff_max < 1e-10);
                
                [alpha_child, a_child, sigma_child] = obj.nodes(id_root).calc_alpha_child(alpha_log, b_log_siblings_pre, m);
                obj.nodes(id_child(m)).set_alpha(alpha_child, sigma_child);
                obj.nodes(id_child(m)).set_a(a_child);
                
                if recursive
                    % recursively calculate beta of the children
                    obj.alpha_pass(id_child(m), recursive);
                end
            end
        end
    end
    function cpd_pass( obj, id_root, recursive )
        % this function updates the contitional probability distributions
        % of the current node; it assumes that the a and beta of the 
        % children have already been calculated.
        id_child = obj.nodes(id_root).children();
        K = obj.nodes(id_root).get_K();
        M = size(id_child,1);
        if M > 0
            % if node is a leaf, then there is nothing to do
            for m = 1:M
                % for each child:
                beta_child = obj.nodes(id_child(m)).get_beta();
                a_child = obj.nodes(id_child(m)).get_a();
                obj.nodes(id_root).update_cpd(beta_child, a_child, m);
                
                if recursive
                    % recursively calculate cpd of the children
                    obj.cpd_pass(id_child(m), recursive);
                end
            end
        end
    end
    function lklhd = update_prior( obj, id_root )
        % this function updates the prior (saved in alpha) of the root
        % node; it assumes that the current alpha and beta are available
        % it returns the data log-likelihood for the tree with root id_root
        lklhd = obj.nodes(id_root).update_prior();
    end
    function lklhd = calc_lklhd_per_sample( obj )
        p = parents(obj);
        roots = find(p == 0);
        lklhd = 0;
        for r = 1:length(roots)
            id_root = roots(r);
            [~, lklhd_act] = calc_q(obj.nodes(id_root));
            lklhd = lklhd + lklhd_act;
        end
    end
    function lklhd = calc_lklhd( obj, id_root )
        if get_K(obj, id_root) > 1
            [~, lklhd] = calc_q(obj.nodes(id_root));
            lklhd = nanmean(lklhd,1);
        else
            lklhd = calc_lklhd_child( obj, obj.nodes_parent(id_root), id_root );
        end
    end
    function lklhd = get_lklhd( obj )
        p = parents(obj);
        roots = find(p == 0);
        lklhd = 0;
        for r = 1:length(roots)
            lklhd = lklhd + obj.nodes_lklhd(roots(r));
        end
    end
    function lklhd_child = calc_lklhd_child( obj, id_parent, id_child )
        % different way of calculating the conditional likelihood by marginalizing out
        % the parent;
        % this also works for GMM leaves
        id_children = children(obj.nodes(id_parent));
        m = find(id_children == id_child);
        % check that id_child is actually child of id_parent
        assert(numel(m) == 1);
        [beta_child, beta_child_log_const] = obj.nodes(id_child).get_beta();
        [log_b] = calc_log_b(obj.nodes(id_parent), beta_child, m);
        a_child = get_a(obj.nodes(id_child));
        ab_log = log_b + log(a_child);
        [ q_parent, lklhd_child ] = normalize_convex_log( ab_log, 2 );
        % add the log_const
        lklhd_child = lklhd_child + beta_child_log_const;
        % get the mean over all datapoints
        lklhd_child = nanmean(lklhd_child,1);
    end
    function lklhd_child = calc_lklhd_child_all( obj, id_parent)
        % calculate conditional likelihoods for all children of id_parent
        id_child = children(obj.nodes(id_parent));
        M = numel(id_child);
        lklhd_child = zeros(M,1);
        for m = 1:M
            if get_K(obj,id_child(m)) > 1
                % discrete node, we can use calc_lklhd
                lklhd_child(m) = calc_lklhd( obj, id_child(m) );
            else
                % continuous node, we need to use calc_lklhd_child
                lklhd_child(m) = calc_lklhd_child( obj, id_parent, id_child(m) );
            end
        end
    end
    function plot_distrib(obj, id_node, m_child)
        id_child = obj.nodes(id_node).children();
        beta_child = obj.nodes(id_child(m_child)).get_beta();
        obj.nodes(id_node).plot(beta_child, m_child);
    end
    function lklhd_observed = calc_lklhd_observed( obj )
        p = parents(obj);
        lklhd_observed = NaN(obj.M,1);
        for n = 1:obj.M
            if (p(n) == 0) || obj.get_K(n) > 1
                % we can calculate the likelihood directly
                lklhd_observed(n) = calc_lklhd( obj, n );
            else
                % we need to use the parent
                lklhd_observed(n) = calc_lklhd_child( obj, p(n), n );
            end
        end
    end
    function lklhd = EM(obj, id_root, recursive)
        if ~exist('recursive','var') || isempty(recursive)
            recursive = true;
        end
        
        p = parents(obj);
        is_root = p(id_root)==0;
            
        obj.beta_pass(id_root, recursive);
        obj.alpha_pass(id_root, recursive);
        lklhd = calc_lklhd(obj, id_root);
        disp(['start lklhd = ' num2str(lklhd) '; recursive = ' num2str(recursive) '; id_root = ' num2str(id_root) '; num_roots = ' num2str(sum(p==0))]);
        if obj.plot_EM
            plot_all = [obj.plot_graph obj.plot_lklhd_diff obj.plot_EM];
            num_plots = sum(plot_all);
            act_plot_pos = sum(plot_all(1:3));
            if num_plots > 1
                subplot(1,num_plots,act_plot_pos,'replace');
            end
            obj.plot_distrib(id_root,1);
            drawnow;
        end
        lklhd_old = lklhd;
        count = 1;
        continue_condition = true;
        while continue_condition
            if is_root
                % only update the prior if id_node is a root!
                obj.update_prior(id_root);
            end
            obj.cpd_pass(id_root, recursive);
            obj.beta_pass(id_root, recursive);
            obj.alpha_pass(id_root, recursive);
            lklhd = calc_lklhd(obj, id_root);
            diff = lklhd - lklhd_old;
            condition_diff = diff > obj.mindiff;
            condition_count = count < obj.maxcount;
            continue_condition = condition_diff && condition_count;
            if mod(count, obj.dispevery)==0 || ~continue_condition
                disp(['count = ' num2str(count) '; lklhd = ' num2str(lklhd) '; diff = ' num2str(diff)]);
                
                if obj.plot_EM
                    obj.plot_distrib(id_root,1);
                    drawnow;
                end
            end
            if ~condition_diff
               disp(['Stopping because diff not greater than maxdiff = ' num2str(obj.mindiff)]);
            end
            if ~condition_count
                disp(['Stopping because maxcount = ' num2str(obj.maxcount) ' reached.']);
            end
            lklhd_old = lklhd;
            count = count + 1;
        end
    end
    function lklhd = EM_restart(obj, id_root)
        restarts = obj.EM_restarts;
        recursive = false;
        
        if restarts == 1
            lklhd = EM(obj, id_root, recursive);
            return;
        end
        obj.beta_pass(id_root, recursive);
        obj.alpha_pass(id_root, recursive);
        lklhd_best = calc_lklhd(obj, id_root);
        node_best = copy(obj.nodes(id_root));
        lklhd_hist(1) = lklhd_best;
        
        for r = 1:restarts
            disp(['restart ' num2str(r) ':']);
            lklhd = EM(obj, id_root, recursive);
            lklhd_hist(r+1) = lklhd;
            if lklhd > lklhd_best
                lklhd_best = lklhd;
                node_best = copy(obj.nodes(id_root));
            end
            % only initialize the second restart (first try the initial)
            init_node_distribution( obj, id_root, []);
        end
        
        lklhd = lklhd_best;
        obj.nodes(id_root) = node_best;
        obj.beta_pass(id_root, recursive);
        obj.alpha_pass(id_root, recursive);
    end
    
    obj = initialize_algo( obj );
    obj = initialize_parents( obj );
    
    % add root as child of existing parent
    [case1_success, id_changed_node] = training_case1( obj, id_parent_original, id_child_new );  
    % create new node with two roots as children
    [case2_success, id_changed_node] = training_case2( obj, id_parent, id_child_new );
    % create new node between an existing parent and an existing child
    [case3_success, id_changed_node] = training_case3( obj, id_parent, id_child_new );
    
    obj = training( obj, X_train, K, X_validation);
    function [lklhd, inf_dist, inf_dist_cramer, correlation] = calc_potential_lklhd( obj, id_potential_parent, id_child )
        % same as calc_lklhd(), but we try out id_potential_parent as
        % parent for all other (root) nodes and then get the posterior
        % likelihood
        
        % we need:
        % q(potential_parent)
        % beta_child
        q_pp = calc_q(obj.nodes(id_potential_parent));
        % q_pp: N x K_parent
        M = length(id_child);
        lklhd = NaN(M,1);
        inf_dist = NaN(M,1); %information distance
        inf_dist_cramer = NaN(M,1);
        correlation = NaN(M,1);
        for m = 1:M
            [beta_child, beta_child_log_const] = obj.nodes(id_child(m)).get_beta(); 
            % beta_root: N x K_child
            [Mu, Sigma] = hpmm2.calc_MuSigma(q_pp, beta_child);
            [log_b] = hpmm2.calc_log_b(beta_child, Mu, Sigma);
            log_b = bsxfun(@plus, log_b, beta_child_log_const);
            % log_b: N x K_parent
            
            % inner product between q_pp and b:
            log_q_pp = log(q_pp);
            log_q_pp = hpmm2.set_nans_zero(log_q_pp);
            log_bq = log_b + log_q_pp;
            c = max(log_bq,[],2);
            bq = exp(bsxfun(@minus, log_bq, c));
            log_bq_sum = log(sum(bq,2)) + c;
            % log_p = p( x_in(child) | x_out(child) )
            log_p = nanmean(log_bq_sum,1);
            lklhd(m) = log_p;
            
            % get correlation / MI between prediction and log_b
            % a_child is q in this case
%             alpha_child = hpmm2.calc_alpha(q_pp, Mu);
%             K_child = size(Mu,2);
            
%             if K_child == 1
%                 inf_dist(m) = -log(abs(corr(alpha_child, beta_child)));
%                 inf_dist_cramer(m) = inf_dist(m);
%                 correlation(m) = corr(alpha_child, beta_child);
%             else
%                 inf_dist(m) = information_distance_discrete(alpha_child, beta_child);
%                 inf_dist_cramer(m) = -log(CramersV_discrete(alpha_child, beta_child));
%                 
%                 exp_beta = beta_child * (1:K_child)';
%                 exp_alpha = alpha_child * (1:K_child)';
%                 correlation(m) = corr(exp_beta, exp_alpha);
%             end
        end
    end
    function obj = set_potential_lklhd( obj, act_top, act_parent, act_child )
        [lklhd, inf_dist, inf_dist_cramer, correlation] = calc_potential_lklhd( obj, act_parent, act_child );
        
        obj.nodes_lklhd_diff(act_top, act_child ) = (lklhd - obj.nodes_lklhd(act_child))';
        obj.nodes_inf_dist(act_top, act_child ) = inf_dist;
        obj.nodes_inf_dist_cramer(act_top, act_child ) = inf_dist_cramer;
        obj.nodes_corr(act_top, act_child ) = correlation;
        
        obj.nodes_lklhd_diff(act_parent, act_child ) = (lklhd - obj.nodes_lklhd(act_child))';
        obj.nodes_inf_dist(act_parent, act_child ) = inf_dist;
        obj.nodes_inf_dist_cramer(act_parent, act_child ) = inf_dist_cramer;
        obj.nodes_corr(act_parent, act_child ) = correlation;
    end
    obj = set_potential_lklhd_changed( obj, id_changed_node );
    [id_parent, id_child, next, both_root] = find_best_nodes_to_merge( obj );
    function obj = normalize_beta(obj, id_parent)
        % normalize beta values, need ed for local likelihood of the root
        id_child = obj.children(id_parent);
        for c = 1:length(id_child)
            obj.nodes(id_child(c)).normalize_beta();
        end
    end
    
    function obj = inference( obj, X_o, ind_o )
       % utility function for doing inference on new data X_o
       
       % clear old data first
        obj = clear_data( obj );
        % insert the data into the tree
        obj = set_data(obj, X_o, ind_o);

        % do forward-backward over all roots
        p = parents( obj );
        id_root = find(p==0);
        for r = 1:length(id_root)
            beta_pass( obj, id_root(r), true );
            alpha_pass( obj, id_root(r), true )
        end
    end
    function [X_u, Var_u, lklhd] = testing( obj, X_o, ind_o, ind_u )
        N_test = hpmm2.data_get_NM(X_o);
        N_max = 5000;
        if N_test > N_max
            [~, num_repeat] = serialize_indices(N_test, N_max);
%             num_repeat = ceil(N_test / N_max);
%             N_start = (0:(num_repeat-1)) .* N_max + 1;
%             N_finish = (1:num_repeat) .* N_max;
%             N_finish(end) = N_test;
            
            X_u = zeros(N_test, length(ind_u));
            Var_u = zeros(N_test, length(ind_u));
            lklhd = zeros(N_test, 1);
            for r = 1:num_repeat
                % idx = N_start(r):N_finish(r);
                idx = serialize_indices(N_test, N_max, r);
                [X_u(idx,:), Var_u(idx,:), lklhd(idx,:)] = testing( obj, hpmm2.data_get_rows(X_o, idx), ind_o, ind_u );
            end
        else
            obj = inference( obj, X_o, ind_o );
            lklhd = calc_lklhd_per_sample( obj );
            % get the unobserved dimensions out of the tree
            [X_u, Var_u] = get_data(obj, ind_u);
        end
    end
    
    % stats functions
    function num = num_children(obj)
        L = length(obj.nodes);
        num = zeros(L,1);
        for l = 1:L
            num(l) = length(obj.children(l));
        end
    end
    
    % nodes_variables update functions
    obj = nodes_case1(obj, id_parent_original, id_child_new);
    obj = nodes_case2(obj, id_parent, id_child_new, id_new_node);
    obj = nodes_case3(obj, id_parent, id_child_new, id_new_node);
    
    %
    function [obj, id_closed_new] = nodes_update_closed(obj)
        is_open = ~obj.nodes_closed & obj.nodes_top;
        open = find(is_open);
        
        [~, is_unobserved] = get_observed_nodes(obj);
        
        id_closed_new = [];
        
        for n = 1:length(open)
            scores_neg = (obj.nodes_lklhd_diff(open(n),:) <= obj.minscore)' | (obj.nodes_lklhd_diff(:,open(n)) <= obj.minscore);
            scores_neg(open(n)) = true; % selfscore is always negative
            any_failed = any(obj.nodes_failed(open(n),:,:),3)';
            
            is_closed = scores_neg | is_open | any_failed | ~obj.nodes_top;
            if obj.lower_layers_first
                higher_layer = obj.nodes_layer > obj.nodes_layer(open(n));
                is_closed = is_closed | higher_layer;
            end
            if obj.combine_observed_first
                is_closed = is_closed | is_unobserved;
            end
            is_closed = all(is_closed);
            
            if is_closed
                % node is closed if for all other top nodes at least one of
                % the following is true:
                % the score is negative OR the node is open OR the
                % connection has already been tried out and failed OR the
                % node is not on top
                obj.nodes_closed(open(n)) = true;
                obj.nodes_layer(open(n)) = obj.nodes_layer(open(n)) + 1;
                id_closed_new = cat(1, id_closed_new, open(n));
            end
        end
        
        if obj.recursuve_EM_after_new_closed
            for i = 1:length(id_closed_new)
                lklhd_root = EM(obj, id_closed_new(i), true);
                obj.nodes_lklhd(id_closed_new(i)) = lklhd_root;
            end
            for i = 1:length(id_closed_new)
                obj.set_potential_lklhd_changed( id_closed_new(i) );
            end
        end
        
    end
    
    function [is_observed, is_unobserved] = get_observed_nodes(obj)
        is_observed = true(size(obj.nodes_top));
        is_unobserved = false(size(obj.nodes_top));
        is_observed(obj.ind_u) = false;
        is_unobserved(obj.ind_u) = true;
        parent_u = obj.nodes_parent(obj.ind_u);
        parent_u(parent_u == 0) = [];
        is_observed(parent_u) = false;
        is_unobserved(parent_u) = true;
    end
    function obj = history_update( obj )
        if obj.history_keep
            num_param = get_num_parameters_total( obj );
            lklhd = get_lklhd( obj );
            BIC = - 2 * obj.N * lklhd + num_param * (log(obj.N) - log(2*pi));
            num_hidden = size(obj.nodes,1) - obj.M ;
            hist_row = [lklhd num_param BIC num_hidden];
            obj.history = cat(1, obj.history, hist_row);
        end
    end
    
    function [alpha, sigma] = get_alpha(obj, id_node)
        num_nodes = numel(id_node);
        alpha = cell(1,num_nodes);
        sigma = cell(1,num_nodes);
        for i = 1:num_nodes
            [alpha_act, sigma_act] = get_alpha(obj.nodes(id_node(i)));
            % check for case that alpha is only the prior: in that case
            % repeat the prior for each data point
            if size(alpha_act,1) == 1
                alpha_act = repmat(alpha_act, [obj.N 1]);
                sigma_act = repmat(sigma_act, [obj.N 1]);
            end
            
            alpha{i} = alpha_act;
            sigma{i} = sigma_act;
        end
    end
    
    function result = copy_all(obj)
       result = copy(obj);  % only shallow copy 
       for n = 1:length(obj.nodes)
           result.nodes(n) = copy(obj.nodes(n));
       end
    end
end % methods
end % classdef
