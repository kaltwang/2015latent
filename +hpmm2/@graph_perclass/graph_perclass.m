
classdef graph_perclass

properties
    % Parameters
    ind_class = [];
    K_class = [];
    graph = hpmm2.graph.empty();
    graphs_perclass = hpmm2.graph.empty();
end

methods
    % constructor
    function obj = graph_perclass(varargin)               
        obj = set(obj,varargin{:});
    end
    
    function [ obj ] = set( obj, varargin )
        if rem(nargin-1,2)
            error('Arguments to set should be (property, value) pairs')
        end
        numSettings	= (nargin-1)/2;
        for n = 1:numSettings
            property	= varargin{(n-1)*2+1};
            value		= varargin{(n-1)*2+2};
            obj.(property) = value;
        end
    end
    
    function obj = training( obj, X, K)
       % first train graph on all data, but independent from the selected class
       N = size(X,1);
       K = shiftdim(K)';  % K: 1 x M
       [X, X_class] = split_idx(X, obj.ind_class);
       [K, K_class_] = split_idx(K, obj.ind_class);
       assert(K_class_ > 1)  % not possible for continuous dimensions
       obj.graph = training( obj.graph, X, K');
       
       % clear the data before copying
       clear_data(obj.graph);
       % then, optimize the weights separately for the different classes
       num_used = 0;
       for k = 1:K_class_
           % copy the graph
           graph_act = copy_all(obj.graph);
           
           % prepare data
           [X_act, num_act] = selct_class(X, X_class, k);
           num_used = num_used + num_act;
           
           graph_act = train_perclass(graph_act, X_act);
           
           obj.graphs_perclass(k) = graph_act;
       end
       % assert(num_used == N); if some values are NaN, then we don't use
       % them at all
       obj.K_class = K_class_;
    end
    
    function [X_u, Var_u, lklhd] = testing( obj, X_o, ind_o, ind_u )
        % two cases: ind_class is observed or not
        N = size(X_o,1);
        class_o = ismember(obj.ind_class, ind_o);
        class_u = ismember(obj.ind_class, ind_u);
        [X_o, ind_o, ind_u, X_class, idx_uc] = map_perclass(obj, X_o, ind_o, ind_u);
        
        lklhd = zeros(N, obj.K_class);
        X_uk = cell(1, obj.K_class);
        Var_uk = cell(1, obj.K_class);
        for k = 1:obj.K_class
            [X_uk{k}, Var_uk{k}, lklhd(:,k)] = testing( obj.graphs_perclass(k), X_o, ind_o, ind_u );
        end
        
        if class_o
            assert(~class_u);
            % set the lklhd of the observed class to inf (this is 
            % equivalent to selcting this class)
            idx_class = sub2ind(size(lklhd), 1:N, X_class');
            lklhd(idx_class) = inf;
        end
        
        % select the X_u and Var_u values of the maximum lklhd
        [~, X_class] = max(lklhd, [], 2);
        M_u = numel(ind_u);
        X_u = zeros(N, M_u);
        Var_u = zeros(N, M_u);
        for k = 1:obj.K_class
            idx_act = X_class == k;
            X_u(idx_act,:) = X_uk{k}(idx_act,:);
            Var_u(idx_act,:) = Var_uk{k}(idx_act,:);
        end
        
        if class_u            
           % we need the inverse of the split oberation
           [X_u] = split_idx_inv(X_u, X_class, idx_uc);
        end
        
    end
    
    function [X_o, ind_o, ind_u, X_class, idx_uc] = map_perclass(obj, X_o, ind_o, ind_u)
        [ind_o, idx_oc] = map_ind(obj, ind_o);
        [X_o, X_class] = split_idx(X_o, idx_oc);
        
        [ind_u, idx_uc] = map_ind(obj, ind_u);
        % no overlap between ind_o and ind_u allowed!
        if any(idx_oc)
            assert(~any(idx_uc));
        end
        if any(idx_uc)
            assert(~any(idx_oc));
        end
        idx_uc = find(idx_uc);
    end
    
    function [ind, idx_c] = map_ind(obj, ind)
        idx_c = ind == obj.ind_class;
        idx_greater = ind > obj.ind_class;
        ind(idx_greater) = ind(idx_greater) - 1;
        ind(idx_c) = [];
    end    
end % methods
end % classdef

function [X, X_sep] = split_idx(X, ind)
    X_sep = X(:,ind);
    X(:,ind) = [];
end

function [X] = split_idx_inv(X, X_sep, ind)
    X1 = X(:, 1:(ind-1));
    X2 = X(:, ind:end);    
    X = [X1 X_sep X2];
end

function [X, num] = selct_class(X, X_class, class)
    idx = X_class == class;
    num = sum(idx);
    X = X(idx,:);
end

function obj = train_perclass(obj, X)
    M = size(X,2);
    id_node = 1:M; % all the input data is set and should correspond to the first nodes
    set_data( obj, X, id_node );
    
    p = parents(obj);
    roots = find(p == 0);
    lklhd = 0;
    recursive = true;
    for r = 1:length(roots)
        id_root = roots(r);
        lklhd = lklhd + EM(obj, id_root, recursive);
    end
end