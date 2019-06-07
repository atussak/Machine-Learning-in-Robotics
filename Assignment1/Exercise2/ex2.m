images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

d = 3;

%% (1) Make the training data zero mean

% Mean of each pixel/row ( --> a probably very strange mean image)
images = images - mean(images, 2);


%% (2) Calculate the covariance

covariance = cov(images');

%% (3) Calculate eigenvalues and eigenvectors

[eigvec, eigval] = eig(covariance);

%% (4) Choose d eigenvectors with largest eigenvalues
    % = the principal components of the data

eigval_1d = diag(eigval);
[max_eigvals, max_idxs] = maxk(eigval_1d',d);
principal_comps = []; 
for i = max_idxs
   principal_comps = [principal_comps eigvec(:,i)]; 
end