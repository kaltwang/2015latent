function id_changed_node = nodes_case3( obj, id_parent, id_child_new, id_new_node )
% case 3:
% create new node between an existing parent and an existing child
% create new node as child of id_parent and add id_child as a
% child of the new node (parent -> new_node ->  id_child new)

% id_new_node is not on top
obj.nodes_top(id_new_node) = false;
obj.nodes_parent(id_new_node) = id_new_node;

% Id_parent cannot be on top and cannot be virtual node and cannot have a
% virtual parent
assert(obj.nodes_top(id_parent) == false);
assert(obj.nodes_parent(id_parent) == id_parent);
% Id_child_new must be on top and cannot be virtual
assert(obj.nodes_top(id_child_new) == true);
assert(obj.nodes_parent(id_child_new) ~= 0);

% afterwards:
% remove id_child_new from top
obj.nodes_top(id_child_new) = false;
% parent status of child_id_new stays the same
% top status of id_parent stays the same
% parent status of id-parent stays the same

% just to check
id_child_old = children(obj, id_parent);
assert(all(obj.nodes_top(id_child_old) == false));

id_child_new_parent = obj.nodes_parent(id_child_new);
if id_child_new ~= obj.nodes_parent(id_child_new)
    % virtual has no parent
    assert(obj.nodes_parent(id_child_new_parent) == 0);
    % parent must not be on top
    assert(obj.nodes_top(id_child_new_parent) == false);

    % afterwards:
    % id_child_new becomes 'virtual', since it is not
    % considered anymore
    obj.nodes_parent(id_child_new) = 0;

    % we need to delete the virtual parent of id_child_new
    delete_id = id_child_new_parent;
else
    delete_id = [];
end



% update tree_ids
tree_ids = obj.nodes_tree_id([id_parent id_child_new]);
tree_id_new = tree_ids(1);
idx = (obj.nodes_tree_id == tree_ids(1)) | (obj.nodes_tree_id == tree_ids(2));
obj.nodes_tree_id(idx) = tree_id_new;
obj.nodes_tree_id(id_new_node) = tree_id_new;

% initialize nodes_failed
obj.nodes_failed(id_new_node,id_new_node,[1 2 3]) = false;

% Warning: node_ids changed after deletion!
if ~isempty(delete_id)
    obj.delete_node(delete_id);
end

% for further EM steps, id_parent is the root
id_changed_node = id_parent;
if id_changed_node >= delete_id;
    % reindex after node deletion
    id_changed_node = id_changed_node - 1;
end
end