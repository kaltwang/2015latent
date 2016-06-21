function [case2_success, id_changed_node] = training_case2( obj, id_parent, id_child_new )
% case 2:
% create new node with two roots as children
% also fall back to case 2 if case 1 has not brought any
% improvement for the partial conditional likelihood

%lklhd_cond_old = [obj.nodes_lklhd(id_parent); obj.nodes_lklhd(id_child_new)];
lklhd_cond_old = [obj.calc_lklhd(id_parent); obj.calc_lklhd(id_child_new)];

if ~isempty(obj.focus_dim)
    focus = obj.nodes_focus([id_parent; id_child_new]);
else
    focus = true(length(lklhd_cond_old),1);
end

% make copies
parent_copy = copy(obj.nodes(id_parent));
child_copy = copy(obj.nodes(id_child_new));

% add new parent
id_new_parent = add_node( obj, [id_parent; id_child_new], obj.K_default );
disp(['Case 2: Add new node ' num2str(id_new_parent) ' with children (' num2str(id_parent) ',' num2str(id_child_new) ')']);
% lklhd_root = node_find_K( obj, id_new_parent);
init_node_distribution( obj, id_new_parent, obj.K_default );
lklhd_root = EM_restart(obj, id_new_parent);

lklhd_cond_new = calc_lklhd_child_all( obj, id_new_parent);

conditional_diff = lklhd_cond_new - lklhd_cond_old;
check_conditional = conditional_diff(focus) > 0;
check_joint = (lklhd_root - sum(lklhd_cond_old)) > 0;
case2_success = all(check_conditional) && check_joint;

if ~case2_success
    % report what happened
    id_child = children(obj, id_new_parent);
    nodes_str = '';
    if ~check_joint
        nodes_str = [nodes_str ' (' num2str(id_new_parent) ', root!)'];
    end
    for i = 1:numel(check_conditional)
        if ~check_conditional(i)
            nodes_str = [nodes_str ' (' num2str(id_new_parent) ',' num2str(id_child(i)) ')'];
        end
    end
    disp(['XXXXXXXXXXXXXX Case 2: Decrease in likelihoods' nodes_str '!']);
    disp('Fall back to case 2 or 3.');

    % restore old parent
    obj.nodes(id_parent) = parent_copy;
    obj.nodes(id_child_new) = child_copy;
    
    % delete the new node
    delete(obj.nodes(id_new_parent));
    obj.nodes(id_new_parent) = [];

    id_changed_node = [];

    % mark failed
    obj.nodes_failed(id_parent, id_child_new, 2) = true;
else
    
    [id_changed_node] = nodes_case2( obj, id_parent, id_child_new, id_new_parent );
    obj.node_changed_last = [];
end

end