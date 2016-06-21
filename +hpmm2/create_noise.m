function [X] = create_noise(X)

if ~iscell(X)
    X = randn(size(X));
else
    num_cell = numel(X);
    for i = 1:num_cell
        X_act = X{i};
        K = size(X_act, 2);
        if K == 1
            % continuous data
            X_act = randn(size(X_act));
        else
            % discrete data
            X_act = rand(size(X_act));
            X_act = normalize_convex(X_act,2);
        end
        X{i} = X_act;
    end
end

end