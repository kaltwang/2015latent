function obj = set_potential_lklhd_changed( obj, id_changed_node )

switch obj.strategy
    case 'always combine'
        % get lklhd_diff for new top node
        % nodes_lklhd_diff
        % nodes_inf_dist
        % nodes_inf_dist_cramer
        id_all = find(obj.nodes_parent ~= 0);
        id_all_parent = obj.nodes_parent(id_all);
        % first, re-calculate all the likelihoods
        % only nodes within the same tree as the changed root need to
        % be renewed
        id_all_changed = find(obj.nodes_parent ~= 0 & obj.nodes_tree_id == obj.nodes_tree_id(id_changed_node));
        % none of the nodes can be virtual
        assert(all(id_all_changed == obj.nodes_parent(id_all_changed)));
        for t = 1:length(id_all_changed)
            obj.nodes_lklhd(id_all_changed(t)) = calc_lklhd( obj, id_all_changed(t) );
        end

        % get the dist old_top -> id_changed_node
        for t = 1:length(id_all_parent)
            % act_child = id_changed_node;
            act_child = (1:length(obj.nodes))';
            act_top = id_all(t);
            act_parent = id_all_parent(t);
            set_potential_lklhd( obj, act_top, act_parent, act_child );
        end
    case {'combine closed', 'combine roots with new hidden'}
        id_all = find(obj.nodes_top);
        id_all_parent = obj.nodes_parent(id_all);
        % we have already all the likelihoods of the roots,
        % no need for recalculation

        % get the dist top -> id_changed_node
        for t = 1:length(id_all_parent)
            act_child = id_changed_node;
            act_top = id_all(t);
            act_parent = id_all_parent(t);
            set_potential_lklhd( obj, act_top, act_parent, act_child );
        end
        if ~isempty(obj.ind_u)
            bin_all = obj.nodes_top;
            bin_all(obj.ind_u) = true;
            id_all = find(bin_all);
        end
        % get the dist id_changed_node -> top
        set_potential_lklhd( obj, id_changed_node, obj.nodes_parent(id_changed_node), id_all );
    case 'combine unobserved'
        [is_observed, is_unobserved] = get_observed_nodes(obj);
        is_observed = is_observed & (obj.nodes_parent ~= 0);
        is_unobserved = is_unobserved & (obj.nodes_parent ~= 0);
        id_o = find(is_observed);
        id_u = find(is_unobserved);
        id_o_parent = obj.nodes_parent(id_o);
        for t = 1:length(id_o_parent)
            act_child = id_u;
            act_top = id_o(t);
            act_parent = id_o_parent(t);
            set_potential_lklhd( obj, act_top, act_parent, act_child );
        end
    otherwise
        error(['Unknown strategy: ' obj.strategy]);
end

end