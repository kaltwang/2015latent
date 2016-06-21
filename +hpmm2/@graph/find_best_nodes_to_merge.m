function [id_parent, id_child, next, both_root] = find_best_nodes_to_merge( obj )
% hardcoded use of nodes_lklhd_diff and score threshold
% TODO: include as graph properties

nodes_all = obj.nodes_parent ~= 0;
% eligible are all pairs that:
% - come from different trees
% - both are real nodes
% - either two roots or 1 root and 1 inner node

is_root = obj.nodes_top;
[is_observed, is_unobserved] = get_observed_nodes(obj);
observed_observed = bsxfun(@and, is_observed, is_observed');
observed_unobserved = bsxfun(@and, is_observed, is_unobserved');
real = bsxfun(@and, nodes_all, nodes_all');
different_trees = bsxfun(@ne, obj.nodes_tree_id, obj.nodes_tree_id');
switch obj.strategy
    case 'combine roots with new hidden'
        root_root = bsxfun(@and, is_root, is_root');
        eligible = root_root;
        
        % remove the diagonal
        eligible(eye(size(eligible))==1) = false;
        
    case 'always combine'
        is_inner = (1:length(obj.nodes_top))' > obj.M;
        root_root = bsxfun(@and, is_root, is_root');
        root_inner = bsxfun(@and, is_root, is_inner');
        root_condition = root_inner | root_inner' | root_root; % it doesnt matter if the first or second one is the inner
        
        eligible = different_trees & real & root_condition;
        
    case 'combine closed'
        % combining either 2 closed nodes with new parent or add 1 closed
        % as a child of an open node.
        % both nodes bust be on top.
        is_closed = obj.nodes_closed & obj.nodes_top;
        is_open = ~obj.nodes_closed & obj.nodes_top;
        closed_closed = bsxfun(@and, is_closed, is_closed');
        open_closed = bsxfun(@and, is_open, is_closed');
        
        % eligible = closed_closed | open_closed | open_closed';
        eligible = closed_closed | open_closed | open_closed';
        
        % remove the diagonal
        eligible(eye(size(eligible))==1) = false;
    case 'combine unobserved'
        eligible = observed_unobserved & real;
        % unobserved must be on top
        eligible = bsxfun(@and, eligible, obj.nodes_top');
        eligible(eye(size(eligible))==1) = false;
    otherwise
        error(['Unknown strategy: ' obj.strategy]);
end

if obj.combine_observed_first
    eligible = eligible & observed_observed & different_trees;
end

% remove too low values
eligible = eligible & obj.nodes_lklhd_diff > obj.minscore;

% remove failed
% it doesnt matter which way around we tried already
is_failed = any(obj.nodes_failed,3);
% switch obj.strategy
%     case 'always combine'
%         is_failed = any(obj.nodes_failed,3);
%     case 'combine closed'
%         idx = zeros(size(closed_closed));
%         idx(closed_closed) = 2;
%         idx(~closed_closed) = 1;
%         is_failed = select_dim(obj.nodes_failed, 3, idx);
% end

failed_failed = bsxfun(@or, is_failed, is_failed');
eligible = eligible & ~failed_failed;

if ~isempty(obj.focus_dim)
    is_focus = bsxfun(@or, obj.nodes_focus, obj.nodes_focus');
    eligible = eligible & is_focus;
end

if obj.close_last_first
    if ~isempty(obj.node_changed_last) && ~obj.nodes_closed(obj.node_changed_last)
        % close the last updated node first
        is_last = false(size(eligible,1),1);
        is_last(obj.node_changed_last) = true;
        eligible = eligible & bsxfun(@or, is_last, is_last');
    end
end

if obj.lower_layers_first
    layer_max = bsxfun(@max, obj.nodes_layer, obj.nodes_layer');
    layer_min_eligible = min(layer_max(eligible(:)));
    if ~isempty(layer_min_eligible)
        eligible = eligible & layer_max == layer_min_eligible;
    end
end



%         top = find(obj.nodes_top);        
scores = obj.nodes_lklhd_diff;

%         % exclude own scores (on the diagonal)
%         scores(eye(size(scores))==1) = -Inf;
% exclude non-eligible
scores(~eligible) = -Inf;

[m1, id1] = max(scores,[],1);
[m12, id2] = max(m1,[],2);
id1 = id1(id2);

%         id_parent = obj.nodes_parent(top(id1));
%         id_child = top(id2);
id_parent = id1;
id_child = id2;
score = m12;

both_root = is_root(id_parent) & is_root(id_child);
switch obj.strategy
    case {'combine roots with new hidden', 'combine unobserved'}
        % no switch needed
        need_switch = false;
    case 'always combine'
        need_switch = ~both_root && is_root(id_parent);
    case 'combine closed'
        if obj.close_last_first
            if obj.nodes_layer(id_child) > obj.nodes_layer(id_parent)
                need_switch = true;
            else
                if obj.nodes_layer(id_child) < obj.nodes_layer(id_parent)
                    need_switch = false;
                else
                    need_switch = ~obj.nodes_closed(id_child);
                end
            end
        else
            need_switch = ~obj.nodes_closed(id_child);
        end
    otherwise
        error(['Unknown strategy: ' obj.strategy]);
end



if need_switch
    % we need to switch parent and child
    % since parent needs to be the inner node
    tmp = id_parent;
    id_parent = id_child;
    id_child = tmp;
end

next = score > obj.minscore;

if ~next && obj.combine_observed_first
    % in this case, we can include the unobserved
    obj.combine_observed_first = false;
    % change the strategy
    obj.strategy = 'combine unobserved';
    obj.recursuve_EM_final = true;
    obj.lower_layers_first = false;
    obj.close_last_first = false;
    % calculate all likelihoods
    obj = set_potential_lklhd_changed( obj, [] );
    [id_parent, id_child, next, both_root] = find_best_nodes_to_merge( obj );
    return;
end


if obj.plot_lklhd_diff
    plot_all = [obj.plot_graph obj.plot_lklhd_diff obj.plot_EM];
    num_plots = sum(plot_all);
    act_plot_pos = sum(plot_all(1:2));
    if num_plots > 1
        subplot(1,num_plots,act_plot_pos,'replace');
    end
    % imagesc(scores);
    tmp = obj.nodes_lklhd_diff;
    tmp(eye(size(tmp))==1) = -Inf;
    imagesc(tmp);
end

end