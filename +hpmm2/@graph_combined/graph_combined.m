
classdef graph_combined

properties
    graphs = hpmm2.graph.empty();
    combination_method = 'posterior_per_sample';
    N_max = 5000;
end

methods
    % constructor
    function obj = graph_combined(varargin)               
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
    function [X_u, Var_u] = testing( obj, X_o, ind_o, ind_u )
        N_test = hpmm2.data_get_NM(X_o);
        if N_test > obj.N_max
            % shard over testing samples 
            [~, num_repeat] = serialize_indices(N_test, obj.N_max);            
            X_u = zeros(N_test, numel(ind_u));
            for r = 1:num_repeat
                idx = serialize_indices(N_test, obj.N_max, r);
                X_u(idx,:) = testing( obj, hpmm2.data_get_rows(X_o, idx), ind_o, ind_u );
            end
        else
            [X_u, Var_u] = testing_single( obj, X_o, ind_o, ind_u );
        end
    end
    
    function [X_u, Var_u] = testing_single( obj, X_o, ind_o, ind_u )
        num_graphs = numel(obj.graphs);
        N = hpmm2.data_get_NM(X_o);
        lklhd = zeros(N,num_graphs);
        X_u_all = cell(1,num_graphs);
        Var_u_all = cell(1,num_graphs);
        for i = 1:num_graphs
            graph_act = obj.graphs(i);
            [X_u_act, Var_u_act, lklhd_act] = hpmm2.graph_combined.testing_graph( graph_act, X_o, ind_o, ind_u );
            
            lklhd(:,i) = lklhd_act;
            X_u_all{i} = X_u_act;
            Var_u_all{i} = Var_u_act;
        end
        
        [ weights ] = get_weights( obj, lklhd );
        [X_u, Var_u] = hpmm2.graph_combined.combine_distributions(X_u_all, Var_u_all, weights);
        % choose as method the same as from the first graph
        method = obj.graphs(1).get_data_mode;
        [X_u, Var_u] = hpmm2.distrib_to_value(X_u, Var_u, method);
    end
   
    function [ weights ] = get_weights( obj, lklhd )
        [N, M] = size(lklhd);
        switch obj.combination_method
            case 'posterior_per_sample'
                [ weights ] = normalize_convex_log( lklhd, 2 );
            case 'posterior_over_all_samples'
                lklhd = sum(lklhd,1);
                [ weights ] = normalize_convex_log( lklhd, 2 );
                weights = repmat(weights, N, 1);
            case 'average'
                weights = ones(N,M) / M;
            otherwise
                error(['Unknown combination_method: ' obj.combination_method]);
        end
    end
end % methods

methods(Static)
    function [X_u, Var_u, lklhd] = testing_graph( graph, X_o, ind_o, ind_u )
        graph = inference( graph, X_o, ind_o );
        lklhd = calc_lklhd_per_sample( graph );
        [X_u, Var_u] = get_alpha(graph, ind_u);
    end
    function [X, Var] = combine_distributions(X_all, Var_all, weights)
        s_nodes = size(X_all{1});
        num_nodes = numel(X_all{1});
        X = cell(s_nodes);
        Var = cell(s_nodes);
        for i = 1:num_nodes
            X_act = cellfun(@(x) x{i}, X_all, 'Uni', false);
            Var_act = cellfun(@(x) x{i}, Var_all, 'Uni', false);
            X_act = cat(3, X_act{:});
            Var_act = cat(3, Var_act{:});
            [X{i}, Var{i}] = hpmm2.graph_combined.combine_distribution(X_act, Var_act, weights);
        end
    end
    function [X, Var] = combine_distribution(X_all, Var_all, weights)
        % num = # of distributions to combine
        % X: N x K x num
        % Var: N x K x num
        % weights: N x num with sum(weights,2) == 1
        [N, K, num] = size(X_all);
        weights = reshape(weights, [N 1 num]);
        % weights: N x 1 x num
        X = bsxfun(@times, X_all, weights);
        X = sum(X, 3);
        
        if ~isempty(Var_all)
            % variance within-components
            Var_wc = bsxfun(@times, Var_all, weights);
            Var_wc = sum(Var_wc, 3);
            % variance between-components
            SqDist = bsxfun(@minus, X, X_all).^2;
            Var_bc = bsxfun(@times, SqDist, weights);
            Var_bc = sum(Var_bc, 3);

            Var = Var_wc + Var_bc;
        else
            Var = [];
        end
    end
end % methods(Static)

end % classdef
