function obj = initialize_algo( obj )
    num_nodes = size(obj.nodes,1);
    id_node = (1:num_nodes)';
    K = get_K( obj, id_node );

    % plot the current graph
    if obj.plot_EM || obj.plot_graph            
        figure;
    end
    if obj.plot_EM && obj.plot_graph
        subplot(2,1,1,'replace');
    end
    if obj.plot_graph
        obj.plot(); drawnow;
    end

    % all nodes are on top:
    obj.nodes_top = true(num_nodes,1);
    % initially, all nodes are on top
    obj.nodes_parent = id_node;
    obj.nodes_lklhd = -Inf(num_nodes,1);
    obj.nodes_tree_id = zeros(num_nodes,1);
    obj.nodes_focus = false(num_nodes,1);
    obj.nodes_layer = ones(num_nodes,1);

    % initialize focus
    obj.nodes_focus(obj.focus_dim) = true;
    
    if ~obj.no_new_hidden_nodes
        obj = initialize_parents( obj );
    else
        % add single parent
        id_node_parent = add_node( obj, id_node, obj.K_default );
        init_node_distribution( obj, id_node_parent, obj.K_default );
        lklhd = EM_restart(obj, id_node_parent);

        obj.nodes_top(:) = false;
        obj.nodes_top(id_node_parent) = true;
        obj.nodes_parent(id_node) = id_node_parent;
        obj.nodes_parent(id_node_parent) = 0;
        obj.nodes_tree_id(id_node) = id_node_parent;
        obj.nodes_tree_id(id_node_parent) = id_node_parent;
        obj.nodes_layer(id_node_parent) = 2;
        obj.nodes_focus(id_node_parent) = obj.nodes_focus(id_node(1));
        
        % plot the current graph
        if obj.plot_graph && (num_nodes < 500 || mod(m,100)==0)
            obj.plot(); drawnow;
        end
    end

    %%% Calculate lklhd distances
    % pass in validation data
    if ~isempty(obj.X_validation)
        obj.testing(obj.X_validation, 1:obj.M, []);
    end

    for m = 1:num_nodes
        id_node_parent = obj.nodes_parent(id_node(m));
        lklhd = calc_lklhd( obj, id_node_parent);
        obj.nodes_lklhd(id_node(m)) = lklhd;
        if K(m) == 1
            % we also need to set the lklhd of the parent
            obj.nodes_lklhd(id_node_parent) = lklhd;
        end
    end

    % get all pairwise information distances
    num_nodes = length(obj.nodes);
    all = find(obj.nodes_parent ~= 0);
    parent = obj.nodes_parent(all);
    obj.nodes_lklhd_diff = -Inf(num_nodes);
    obj.nodes_inf_dist = NaN(num_nodes);
    obj.nodes_inf_dist_cramer = NaN(num_nodes);

    for t = 1:length(all)
        act_child = all;
        act_top = all(t);
        act_parent = parent(t);
        set_potential_lklhd( obj, act_top, act_parent, act_child );                
    end

    % pass in training data
    if ~isempty(obj.X_validation)
        obj.testing(obj.X_train, 1:obj.M, []);
    end
    %%%% finished calculating lklhd distances

    % initialize nodes_failed

    obj.nodes_failed = false([num_nodes num_nodes 3]);
    obj.nodes_closed = false(num_nodes,1);
    obj.nodes_closed(1:obj.M) = true;
    obj.nodes_layer = ones(num_nodes,1);
end