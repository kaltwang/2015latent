function [X, Var] = distrib_to_value(alpha, sigma, method)

if iscell(alpha)
    % in this case, repeat the same function for each cell
    M = numel(alpha);
    N = size(alpha{1},1);
    X = zeros(N, M);
    Var = NaN(N, M);
    for m = 1:M
        [X(:,m), Var(:,m)] = hpmm2.distrib_to_value(alpha{m}, sigma{m}, method);
    end
else
    % alpha: N x K with N number of samples and K categories
    % sigma: N x 1 for continuous distributions (K==1), otherwise empty
    % convert categorical distribution to single value

    switch method
        case 'expected value'
            type = 1;
        case 'max value'
            type = 2;
        case 'class1'
            type = 3;
        otherwise
            error(['Unknown conversion method: ' method]);
    end

    [N, K] = size(alpha);
    Var = NaN(N,1);

    if K > 1
        if type == 1
            % in case of discrete nodes, return the expected value
            alpha = alpha * (1:K)';
        else
            if type == 2
            % in case of discrete nodes, return the max value
            [~, alpha_max] = max(alpha,[],2);
            alpha = alpha_max;
            else
                if type == 3
                    % return the probability of the first class
                    alpha = alpha(:,1);
                end
            end
        end
    else
        % only continuous nodes have sigma
        Var = sigma;
    end
    X = alpha;
end
end