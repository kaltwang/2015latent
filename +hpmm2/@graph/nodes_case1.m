function id_changed_node = nodes_case1(obj, id_parent_original, id_child_new)
% id_parent_original can have an virtual node id_parent, in which
% case the id_parent becomes the real parent.
% id_parent cannot be observed leaf
id_parent = obj.nodes_parent(id_parent_original);
if id_parent ~= id_parent_original
    % virtual has no parent
    assert(obj.nodes_parent(id_parent) == 0);
    % original must be on top
    assert(obj.nodes_top(id_parent_original) == true);
    % parent must not be on top
    assert(obj.nodes_top(id_parent) == false);

    % afterwards:
    % id_parent on top
    obj.nodes_top(id_parent) = true;
    % id_parent_original is not on top anymore
    obj.nodes_top(id_parent_original) = false;
    % id_parent becomes 'real' node
    obj.nodes_parent(id_parent) = id_parent;
    % id_parent_original becomes 'virtual', since it is not
    % considered anymore
    obj.nodes_parent(id_parent_original) = 0;
else
    % id_parent can be on top or not
    % this status doesnt change.
    % the parent of id-parent doesnt change either.
end

if ~strcmp(obj.strategy,'combine unobserved')
    % id_child_new must be on top
    % but only if we are not adding the unobserved
    assert(obj.nodes_top(id_child_new) == true);
end

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
% id_child_new is not on top anymore
obj.nodes_top(id_child_new) = false;

% update tree_ids
tree_ids = obj.nodes_tree_id([id_parent id_child_new]);
tree_id_new = tree_ids(1);
idx = (obj.nodes_tree_id == tree_ids(1)) | (obj.nodes_tree_id == tree_ids(2));
obj.nodes_tree_id(idx) = tree_id_new;


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