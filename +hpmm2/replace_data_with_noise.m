function [X] = replace_data_with_noise(X, noise_percent)

[N, M] = hpmm2.data_get_NM(X);
noise_dim = round(M * noise_percent);
if noise_dim == 0
    return;
end

% get random permutations
p = false(N,M);
for i = 1:N
   p(i,randperm(M,noise_dim)) = true;
end

% get noise
X_noise = hpmm2.create_noise(X);

if ~iscell(X)
    X(p) = X_noise(p);
else
    num_cell = size(X,2);
    for i = 1:num_cell
        X{i}(p(:,i),:) = X_noise{i}(p(:,i),:);
    end
end

end