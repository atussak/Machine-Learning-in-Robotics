load("dataGMM.mat");

%% a

k = 4; % number of components in GMM
[d,n] = size(Data);
% Init parameters
[idx, c, sumd] = kmeans(Data', k);

pi     = zeros(1,k);
mu     = zeros(d,k);
sigma  = zeros(d,d,k);

for i = 1:4
    data_i = Data(:, idx==i);
    nk = sum(idx==i);
    
    pi(i) = nk/n;
    mu(:,i) = mean(data_i,2)';
    
    diff = data_i - mu(:,i);
    sigma(:,:,i) = pi(i)*diff*diff'/nk;
end


%% b

