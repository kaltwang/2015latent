function [id_changed_node] = nodes_case2( obj, id_parent, id_child_new, id_new_node )

% add id_new_parent to top
obj.nodes_top(id_new_node) = true;
obj.nodes_parent(id_new_node) = id_new_node; 

% id_parent cannot have a virtual parent (in this case, case1 would have
% been successful)
assert(obj.nodes_parent(id_parent) == id_parent);
if obj.nodes_parent(id_child_new) ~= id_child_new
    % we need to delete this node
    delete_id = obj.nodes_parent(id_child_new);
    % if id_child_new has a virtual parent, then now itself will be virtual
    obj.nodes_parent(id_child_new) = 0;
else
    delete_id = [];
end

% remove id_parent and id_child_new from top
obj.nodes_top(id_parent) = false;
obj.nodes_top(id_child_new) = false;
% double check the children of parent
id_child_old = children(obj, id_parent);
assert(all(obj.nodes_top(id_child_old) == false));

% update tree_ids
tree_ids = obj.nodes_tree_id([id_parent id_child_new]);
tree_id_new = id_new_node;
idx = (obj.nodes_tree_id == tree_ids(1)) | (obj.nodes_tree_id == tree_ids(2));
obj.nodes_tree_id(idx) = tree_id_new;
obj.nodes_tree_id(id_new_node) = tree_id_new;

% initialize nodes_failed
obj.nodes_failed(id_new_node,id_new_node,[1 2 3]) = false;

obj.nodes_closed(id_new_node,1) = false;
obj.nodes_focus(id_new_node,1) = any(obj.nodes_focus([id_parent id_child_new]));
obj.nodes_layer(id_new_node,1) = max(obj.nodes_layer([id_parent id_child_new]));

% Warning: node_ids changed after deletion!
if ~isempty(delete_id)
    obj.delete_node(delete_id);
end

% for further EM steps, id_new_node is the root
id_changed_node = id_new_node;
if id_changed_node >= delete_id;
    % reindex after node deletion
    id_changed_node = id_changed_node - 1;
end
end