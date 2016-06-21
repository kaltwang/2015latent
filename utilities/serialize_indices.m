function [idx, num_folds] = serialize_indices(N_total, N_max, fold)
    if ~exist('fold','var') || isempty(fold)
        fold = 1;
    end
    
    num_folds = ceil(N_total / N_max);
    
    idx = false(N_total,1);
    n_start = (fold-1) * N_max + 1;
    n_finish = (fold) * N_max;
    
    % limit both to [1, N_total]
    n_start = min(max(1, n_start), N_total);
    n_finish = min(max(1, n_finish), N_total);
    
    idx(n_start:n_finish) = true;
end