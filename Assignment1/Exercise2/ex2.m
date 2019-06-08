images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

d = 3;

%% (1) Make the training data zero mean

% Mean of each pixel/row ( --> a probably very strange mean image)
mean_img = mean(images, 2);
images = images - mean_img;


%% (2) Calculate the covariance

covariance = cov(images');

%% (3) Calculate eigenvalues and eigenvectors

[eigvec, eigval] = eig(covariance);

%% (4) Choose d eigenvectors with largest eigenvalues
    % = the principal components of the data
    % = transformation matrix W

eigval_1d = diag(eigval);
[max_eigvals, max_idxs] = maxk(eigval_1d',d);
W = []; 
for i = max_idxs
   W = [W eigvec(:,i)]; 
end

%% (5) Project the training data

% y = W'x
projected_imgs = W'*images; % now the data has a lower dimension!


%% (6) Make test data zero mean and project it on learned basis

test_images = test_images - mean_img;
projected_test_imgs = W'*test_images;

%% (7) Calculate the likelihood of the proj test data for each class

test_sz = size(projected_test_imgs);
test_n = test_sz(2);
likelihood = zeros(10, test_n);

for class = 0:9
    class_idxs = find(labels==class);
    class_imgs = projected_imgs(:,class_idxs);
    class_mean = mean(class_imgs,2);
    class_imgs = class_imgs - class_mean;
    class_cov = cov(class_imgs');
    
    % Multivariate normal probability density function
    likelihood(class+1,:) = mvnpdf(projected_test_imgs', class_mean', class_cov);
end

















