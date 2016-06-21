function [ ] = plot( obj, mapping )
% Plot the graph

plot_all = [obj.plot_graph obj.plot_lklhd_diff obj.plot_EM];
num_plots = sum(plot_all);
act_plot_pos = sum(plot_all(1));
if num_plots > 1
    subplot(1,num_plots,act_plot_pos,'replace');
end

L = size(obj.nodes,1);

parents = obj.parents();

A = obj.get_A();
% labels = [obj.ids() obj.get_K() obj.nodes_layer];
% labels = [obj.ids() obj.get_K()];
labels = [obj.ids()];

if exist('mapping','var')
    nm = numel(mapping);
    map_inv = zeros(nm,1);
    map_inv(mapping) = 1:nm;
    
    % first use the new ids within parents
    idx = parents ~= 0;
    parents(idx) = map_inv(parents(idx));
    % now reorder
    parents = parents(mapping);
    
    labels = map_inv(labels);
    labels = labels(mapping);
    
    A = A(:,mapping);
    A = A(mapping,:);
end

% [x,y,h,s] = treelayout(parents,1:L);
[x,y,h,s] = treelayout(parents);
% % correct ys
% miny = min(y);
% maxy = max(y);
% y = (y - miny) * h /(maxy - miny);
% % correct xs


Coordinates = [x' y'];

% plot
gplot(A,Coordinates,'-*');
% ylim([0 obj.num_layers()+1]);

hold on;
% plot node descriptions
for l = 1:L
    text(Coordinates(l,1),Coordinates(l,2)-0, ...
        [mat2str(labels(l,:)) ' '], ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','middle', ...
        'FontSize',10,...
        'Rotation',90,...
        ... 'BackgroundColor','red',...
        'Color','black');
    if ~isempty(obj.nodes_top) && ~isempty(obj.nodes_closed) && obj.nodes_top(l) && obj.nodes_closed(l)
        plot(Coordinates(l,1),Coordinates(l,2), 'ro');
    end
    if ~isempty(obj.nodes_focus) && obj.nodes_focus(l)
        plot(Coordinates(l,1),Coordinates(l,2), 'g*');
    end
end
hold off;

end