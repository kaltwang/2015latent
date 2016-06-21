function obj = training( obj, X_train, K, X_validation)

if exist('X_validation','var') && ~isempty(X_validation)
    obj.X_validation = X_validation;
    obj.X_train = X_train;
end

init_data( obj, X_train, K);
initialize_algo( obj );

lklhd = get_lklhd( obj );
count = 0;
disp(['(STRUCT) start lklhd = ' num2str(lklhd)]);

obj.history = [];
obj = history_update( obj );

% find best nodes to merge
[id_parent, id_child_new, next, both_root] = find_best_nodes_to_merge( obj );
while next
    lklhd_old = lklhd;
    count = count + 1;

    case1_success = false;
    case2_success = false;
    case3_success = false;

   % both root: either one can be parent or child
   % 1 on top and 1 inner: parent and child are fixed and in case
   % we introduce a new node, then it has to be 
   % parent -> new_node ->  id_child new

   switch obj.strategy
        case 'combine roots with new hidden'
            if obj.nodes_parent(id_parent) ~= id_parent
                % if id_parent has a virtual node, then we will use the
                % virtual one
                [case1_success, id_changed_node] = training_case1( obj, id_parent, id_child_new );
            else
                %otherwise create new node
                [case2_success, id_changed_node] = training_case2( obj, id_parent, id_child_new );
            end
        case 'always combine'
            % case 1: 
            % add id_child to children of id_parent
            % id_parent is NOT observed leaf
            [case1_success, id_changed_node] = training_case1( obj, id_parent, id_child_new );
            %id_remove_virtual = id_child_new;

            % case 1.2:
            if ~case1_success && both_root
               % we can try the other way round
               [case1_success, id_changed_node] = training_case1( obj, id_child_new, id_parent );
               %id_remove_virtual = id_parent;
            end

            % case 2:
            % create new node with id_parent and id_child as children
            % also fall back to case 2 if case 1 has not brought any
            % improvement for the partial conditional likelihood
            if ~case1_success && both_root
               [case2_success, id_changed_node] = training_case2( obj, id_parent, id_child_new );
               %id_remove_virtual = [];
            end

%             % case 3:
%             % create new node as child of id_parent and add id_child as a
%             % child of the new node (parent -> new_node ->  id_child new)
%             if ~case1_success && ~both_root
%                 [case3_success, id_changed_node] = training_case3( obj, id_parent, id_child_new );
%                 %id_remove_virtual = id_child_new;
%             end

        case 'combine closed'

            % only two cases: either on i open and one closed, or
            % both closed
            % id_child_new always has to be closed
            if ~obj.close_last_first
                assert(obj.nodes_closed(id_child_new));
            end

            id_parent_parent = obj.nodes_parent(id_parent);
            if ~obj.nodes_closed(id_parent_parent)
                % case 1: 1 open and one closed
                [case1_success, id_changed_node] = training_case1( obj, id_parent, id_child_new );
            else
                if ~isempty(obj.focus_dim) && obj.no_new_hidden_nodes
                    % hidden nodes with a focused output child are allowed
                    id_nodes = [id_parent; id_child_new];
                    is_focus = obj.nodes_focus(id_nodes);
                    is_observed = id_nodes <= obj.M;
                    if ~any(is_focus & is_observed)
                         % we are done here; don't introduce a new hidden node
                        break;
                    end
                end
                % case 2: both closed; create new node
                [case2_success, id_changed_node] = training_case2( obj, id_parent, id_child_new );
            end
       case 'combine unobserved'
            id_parent_parent = obj.nodes_parent(id_parent);  
            add_child(obj, id_parent_parent, id_child_new );
            id_changed_node = nodes_case1(obj, id_parent, id_child_new);
            case1_success = true;
            % plot the current graph
            if obj.plot_graph
                obj.plot(); drawnow;
            end

            % find best nodes to merge
            [id_parent, id_child_new, next, both_root] = find_best_nodes_to_merge( obj );
            continue;
       otherwise
            error(['Unknown strategy: ' obj.strategy]);
   end

    % this should always be the case, otherwise implement error
    % handling later
    if ~(case1_success || case2_success || case3_success)
      warning(['Step ' num2str(count) ': no success condition reached!']); 
    else

        % update lklhd of changed node
        if obj.recursive_EM_after_struct_change
            lklhd_root = EM(obj, id_changed_node, true);
        end
        
        
        %%% update likelihood distances
        % pass in validation data
        if ~isempty(obj.X_validation)
            obj.testing(obj.X_validation, 1:obj.M, []);
        end
        
        lklhd_root = calc_lklhd(obj, id_changed_node);
        obj.nodes_lklhd(id_changed_node) = lklhd_root;

        obj.set_potential_lklhd_changed( id_changed_node );
        
        % pass in training data
        if ~isempty(obj.X_validation)
            obj.testing(obj.X_train, 1:obj.M, []);
        end
        %%% finished updating likelihood distances
    end
    
    % update the closed nodes
    obj = nodes_update_closed(obj);

    % display actual likelihood:
    lklhd = get_lklhd( obj );
    diff = lklhd - lklhd_old;
    disp(['(STRUCT) count = ' num2str(count) '; lklhd = ' num2str(lklhd) '; diff = ' num2str(diff)]);
    if diff < 0 
       warning(['Step ' num2str(count) ': no likelihood improvement!']); 
    end

    % plot the current graph
    if obj.plot_graph
        obj.plot(); drawnow;
    end

    % find best nodes to merge
    [id_parent, id_child_new, next, both_root] = find_best_nodes_to_merge( obj );
    obj = history_update( obj );
end

if obj.recursuve_EM_final;
    p = obj.parents();
    roots = find(p==0);
    for r = 1:length(roots)
        obj.nodes_lklhd(roots(r)) = EM(obj, roots(r), true);
    end
    obj = history_update( obj );
end

end