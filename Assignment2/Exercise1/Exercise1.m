load("dataGMM.mat");

%% a - init parameters

k = 4; % number of components in GMM
[d,n] = size(Data);

[idx, c] = kmeans(Data', k);

pi     = zeros(1,k);
mu     = zeros(d,k);
sigma  = zeros(d,d,k);

for i = 1:k
    data_i = Data(:, idx==i);
    nk = sum(idx==i);
    
    pi(i) = nk/n;
    mu(:,i) = mean(data_i,2)';
    
    diff = data_i - mu(:,i);
    sigma(:,:,i) = pi(i)*diff*diff'/nk;
end


%% b - EM algorithm

% E-step

gamma = zeros(n,k);
for i = 1:4
    gamma(:,i) = pi(i)*mvnpdf(Data', mu(i), sigma(:,:,i));
end
for j = 1:n
    gamma(j,:) = gamma(j,:)/sum(gamma(j,:));
end

% M-step

for i = 1:k
    nk = sum(gamma(:,i));
    
    pi(i) = nk/n;
    mu(:,i) = gamma(:,i)'*Data'/nk;
    
    diff = Data - mu(:,i);
    sigma(:,:,i) = diff*diag(gamma(:,i))*diff'/nk;
end

% Evaluate

loss = zeros(1,k);
for i = 1:k
    loss(i) = gamma(:,i)'*(log(pi(i))+log(mvnpdf(Data', mu(i), sigma(:,:,i)))-log(gamma(:,i)));
end













