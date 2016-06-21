


% have a look at this script, which describes the model parameters. Some
% need to be cross-validated!
LT_example_model

% Example data X_train with M=10 dimensions and N=100 samples
N = 100;
M = 10;
% size of X_train: N x M
X_train = zeros(N,M);

% K describes the nature of each dimension of the data X: 1 for continuous variables and
% the number of states for discrete variables.
K = zeros(M,1);
% dim 1-8: continuous, gaussian distributed
K(1:8) = 1;
% dim 9: discrete with 3 levels: 1-3
K(9) = 3;
% dim 10: discrete with 5 levels: 1-5
K(10) = 5;

% fill X_train with random values
X_train(:,1:8) = randn(N,8); % random Gaussian
X_train(:,9) = randi([1 3], N, 1); % random integer between 1 and 3
X_train(:,10) = randi([1 5], N, 1); % random integer between 1 and 5


% use the same data as for training, just as example.
X_test = X_train;

% % randomly create 10% missing training features
% n = numel(X_train);
% perc = 0.1; % 10% missing data
% p = randperm(n, round(perc*n));
% X_train(p) = NaN;

% Note: Missing features during training are indicated throug NaNs. Missing
% features during testing should be indicated by ind_o (see the testing 
% example below), i.e. define only the given features as observed.


% the prediction output of my model is a probability distribution. For
% continuous variables, the default output is the expected value of the
% predicted distribution. However, for discrete variables, there are
% several options:
% (a) 'expected value': return the expected value when assuming that the class indices
% correspond to discrete intensities, i.e. class 1 corresponds to intensity
% 1, class 2 corresponds to intensity 2, etc. This makes only sense, if
% there is an order of classes!
% (b) 'max value': return the class index with the highest probability
% (c) 'class1': return the probability of the first class (indexed with 1)
model_hpmm2.get_data_mode = 'max value'; % this is the default, should be fine for most cases.

% train the model.
% training assumes that all dimensions are observed
model_hpmm2 = model_hpmm2.training(X_train, K);

% now test the model.
% assume that only the dimensions [1:7 9] are observed and we try to infer
% dimensions [8 10]
ind_o = [1:7 9]; % this are the indices of the observed dimensions
ind_u = [8 10]; % this are the indices of the unoberved dimensions

% testing gets as input only the observed dimensions (indexed by ind_o)
X_prediction = testing(model_hpmm2, X_test(:,ind_o), ind_o, ind_u);
% the output are only the unobserved dimensions (indexed by ind_u), i.e.
% size of X_prediction is N x 2
