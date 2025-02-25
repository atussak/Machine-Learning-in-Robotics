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
    sigma(:,:,i) = pi(i)*diff*diff'/double(nk);
end

%% b - EM algorithm
converged = false;
conv_criteria = 10e-6;
last_logl = 0;
logl = 1;

while ~ converged
    % E-step    Calculate responsibilities
    gamma = zeros(n,k);
    for i = 1:4
        gamma(:,i) = pi(i)*mvnpdf(Data', mu(:,i)', sigma(:,:,i));
    end
    
    % Normalizing
    gamma = gamma./sum(gamma,2);
    
    % M-step    Re-estimate parameters using the current responsibilities
    for i = 1:k
        nk = sum(gamma(:,i));
        
        pi(i) = nk/n;
        mu(:,i) = gamma(:,i)'*Data'/nk;

        diff = Data - mu(:,i);
        sigma(:,:,i) = diff*diag(gamma(:,i))*diff'/nk;
    end
    
    % Evaluate the log likelihood and check for convergence
    logl = 0;
    for j = 1:n
        temp_loss = 0;
        for i = 1:k
            temp_loss = temp_loss + mvnpdf(Data(:,j)', mu(:,i)', sigma(:,:,i));
        end
        logl = logl + log(temp_loss+10e-20);
    end
    
    conv_test = abs(last_logl-logl);
    if conv_test < conv_criteria
        converged = true;
    end
    last_logl = logl;
end


%% Visualize densities

n_plot = 100;
X = -0.1:0.2/(n_plot-1):0.1;
Y = -0.1:0.2/(n_plot-1):0.1;
data = combvec(X,Y);
Z = zeros(100,100);

for i = 1:k
    Z_temp = mvnpdf(data', mu(:,i)', sigma(:,:,i));
    for r = 0:n_plot-1
       Z(r+1,:) = Z(r+1,:)' + Z_temp(r*n_plot+1:(r+1)*n_plot);
    end
    
end
Z = Z./sum(sum(Z));

figure
density_plot = surf(X,Y,Z);
title('Density values');
% saveas(gcf, strcat("ex1.png"));

