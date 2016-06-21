function obj = initialize_parents( obj )
    num_nodes = size(obj.nodes,1);
    id_node = (1:num_nodes)';
    K = get_K( obj, id_node );

    % Initialize the algorithm:
    % Create GMMs for all continuous observed nodes and calculate the
    % priors for the discrete observed nodes
    for m = 1:num_nodes
        if K(m) == 1
            % continuous dimensions:
            % train single-dim GMM

            % add parent
            id_node_parent = add_node( obj, id_node(m), obj.K_default );
            lklhd = node_find_K( obj, id_node_parent);

            % this node is 'virtual' and therefore not on top (yet)
            obj.nodes_top(id_node_parent) = false;
            obj.nodes_parent(id_node(m)) = id_node_parent;
            obj.nodes_parent(id_node_parent) = 0;
            obj.nodes_tree_id(id_node(m)) = id_node_parent;
            obj.nodes_tree_id(id_node_parent) = id_node_parent;
            obj.nodes_layer(id_node_parent) = 1;
            obj.nodes_focus(id_node_parent) = obj.nodes_focus(id_node(m));

            % plot the current graph
            if obj.plot_graph && (num_nodes < 500 || mod(m,100)==0)
                obj.plot(); drawnow;
            end
        else
            % discrete dimensions:
            % set prior, i.e. alpha
            init_node_distribution( obj, id_node(m) );
            %obj.nodes_lklhd(id_node(m)) = calc_lklhd( obj, id_node(m) ); 
            obj.nodes_tree_id(id_node(m)) = id_node(m);
        end
    end
end