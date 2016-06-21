function [case1_success, id_changed_node] = training_case1( obj, id_parent_original, id_child_new )
% add root as child of existing parent
% id_parent is NOT observed leaf
id_parent = obj.nodes_parent(id_parent_original);    
disp(['Case 1: Try to add edge (' num2str(id_parent) ',' num2str(id_child_new) ')']);

id_child_old = children(obj, id_parent);
if ~isempty(id_child_old)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GET LIKELIHOODS BEFORE
    % get conditional likelihoods of the old children
    lklhd_cond_old = calc_lklhd_child_all( obj, id_parent);
    % on the last position, we put the cond. likelihood of the
    % new child
    %lklhd_cond_old = [lklhd_cond_old; obj.nodes_lklhd(id_child_new)];
    %lklhd_joint_old = obj.nodes_lklhd(id_parent) + obj.nodes_lklhd(id_child_new);
    lklhd_cond_old = [lklhd_cond_old; obj.calc_lklhd(id_child_new)];
    lklhd_joint_old = obj.calc_lklhd(id_parent) + obj.calc_lklhd(id_child_new);
    
    if ~isempty(obj.focus_dim)
        focus = obj.nodes_focus([obj.children(id_parent); id_child_new]);
    else
        focus = true(length(lklhd_cond_old),1);
    end
    
    if obj.lklhd_cond_single
        % check only the first child
        first = find(focus);
        if ~isempty(first)
            focus = false(size(focus));
            focus(first(1)) = true;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MAKE A COPY
    % first, make a copy of the old parent (which might be
    % needed later, if the partial conditional likelihood is
    % decreasing and thus we need to initiate case 2)               
    parent_copy = copy(obj.nodes(id_parent));
    child_copy = copy(obj.nodes(id_child_new));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CHANGE TREE STRUCTURE
    % add child
    add_child(obj, id_parent, id_child_new );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % RUN EM
    if obj.find_best_K
        lklhd_joint_new = node_find_K( obj, id_parent); 
    else
        % lklhd_joint_new = EM(obj, id_parent, false);
        lklhd_joint_new = EM_restart(obj, id_parent);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % GET LIKELIHOODS AFTER
    % check if the partial conditional likelihood got better for
    % each child; the new child should be on the last position
    lklhd_cond_new = calc_lklhd_child_all( obj, id_parent);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EVALUATE LIKELIHOOD DIFFERENCES
    conditional_diff = lklhd_cond_new - lklhd_cond_old;
    check_conditional = conditional_diff(focus) > obj.lklhd_cond_mindiff;
%     if obj.nodes_top(id_parent_original)
%         % only check joint if id_parent_original is on top!
%         check_joint = (lklhd_joint_new - lklhd_joint_old) > 0;
%     else
%         check_joint = true;
%     end
    check_joint = (lklhd_joint_new - lklhd_joint_old) > 0;
    case1_success = all(check_conditional) && check_joint;

    if ~case1_success
        % report what happened
        id_child = children(obj, id_parent);
        nodes_str = '';
        if ~check_joint
            nodes_str = [nodes_str ' (' num2str(id_parent) ', root!)'];
        end
        for i = 1:numel(check_conditional)
            if ~check_conditional(i)
                nodes_str = [nodes_str ' (' num2str(id_parent) ',' num2str(id_child(i)) ')'];
            end
        end
        disp(['XXXXXXXXXXXXXX Case 1 : Decrease in likelihoods' nodes_str '!']);
        disp('Fall back to case 2 or 3.');

        % restore old parent
        obj.nodes(id_parent) = parent_copy;
        obj.nodes(id_child_new) = child_copy;

        id_changed_node = [];
        
        % mark failed
        obj.nodes_failed(id_parent_original, id_child_new, 1) = true;
    else
        
        id_changed_node = nodes_case1(obj, id_parent_original, id_child_new);
        obj.node_changed_last = id_changed_node;
    end
else
    % in this case, we need to create a new node (i.e. case 2 or 3)
    disp('Id_parent is (observed) leaf, go to case 2 or 3.');
    case1_success = false;
    id_changed_node = [];
end
end