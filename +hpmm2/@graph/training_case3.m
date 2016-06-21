function [case3_success, id_changed_node] = training_case3( obj, id_parent, id_child_new )
% case 3:
% create new node between an existing parent and an existing child
% create new node as child of id_parent and add id_child as a
% child of the new node (parent -> new_node ->  id_child new)

% get conditional likelihoods of the old children
lklhd_cond_old = calc_lklhd_child_all( obj, id_parent);
% on the last position, we put the cond. likelihood of the
% new child
lklhd_cond_old = [lklhd_cond_old; obj.nodes_lklhd(id_child_new)];
lklhd_joint_old = obj.calc_lklhd(id_parent) + obj.nodes_lklhd(id_child_new);

if ~isempty(obj.focus_dim)
    focus = obj.nodes_focus([id_parent; id_child_new]);
else
    focus = true(length(lklhd_cond_old),1);
end

% add new node
% new_node ->  id_child new
id_new_node = add_node( obj, id_child_new, obj.K_default );
init_node_distribution( obj, id_new_node, obj.K_default );
beta_pass( obj, id_new_node, false );
% add the new node as a child of parent
% parent -> new_node
add_child(obj, id_parent, id_new_node );

disp(['Case 3: Add new node ' num2str(id_new_node) ' with parent (' num2str(id_parent) ') and child (' num2str(id_child_new) ')']);
%lklhd_root = node_find_K( obj, id_new_node);
% we need to do recursive EM to properly train the new node
% run EM  
% cannot find best_K in this case
lklhd_joint_new = EM(obj, id_parent, true);

% check if the partial conditional likelihood got better for
% each child; 
% the conditional likelihood of the new node doesnt matter;
% the new child should be on the last position
lklhd_cond_new = calc_lklhd_child_all( obj, id_parent);
lklhd_cond_new(end) = calc_lklhd_child_all( obj, id_new_node);

conditional_diff = lklhd_cond_new - lklhd_cond_old;
check_conditional = conditional_diff(focus) > obj.lklhd_cond_mindiff;
check_joint = (lklhd_joint_new - lklhd_joint_old) > 0;
%case3_success = all(check_conditional) && check_joint;
% check only the conditional (joint calculation not properly done!)
case3_success = all(check_conditional);

id_changed_node = nodes_case3( obj, id_parent, id_child_new, id_new_node );
% % add id_new_node is not on top
% obj.nodes_top(id_new_node) = false;
% obj.nodes_parent(id_new_node) = id_new_node;
% 
% % remove id_child_new from top
% id_child_old = children(obj, id_parent);
% obj.nodes_top(id_child_new) = false;
% obj.nodes_top(id_child_old) = false;
% 
% % update tree_ids
% tree_ids = obj.nodes_tree_id([id_parent id_child_new]);
% tree_id_new = tree_ids(1);
% idx = (obj.nodes_tree_id == tree_ids(1)) | (obj.nodes_tree_id == tree_ids(2));
% obj.nodes_tree_id(idx) = tree_id_new;
% obj.nodes_tree_id(id_new_node) = tree_id_new;
% 
% % the changed node is the root of all changed nodes
% id_changed_node = tree_id_new; % (tree_id_new should contain the root id)     
end