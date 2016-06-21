function X = data_get_rows(X, idx)
if ~iscell(X)
    X = X(idx,:);
else
    X = cellfun(@(x) x(idx,:), X, 'Uni', false);
end
end