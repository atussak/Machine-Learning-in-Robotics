load("dataGMM.mat");

%% a

k = 4; % number of components in GMM

[idx, c, sumd] = kmeans(Data', k);