function Exercise3_nubs(gesture_l, gesture_o, gesture_x, K)

titles = ["l-gesture", "o-gesture", "x-gesture"];

% Split vector
v = [0.08, 0.05, 0.02];

gestures = {gesture_l, gesture_o, gesture_x};
num_gests = size(gestures);

for i = 1:num_gests(2)
    % Distortions
    J = zeros(1,K); 
    % Class centers
    centers = zeros(3,K);
    % Data points
    [m,n,o] = size(gestures{i});
    all_data = reshape(gestures{i},[m*n,o]);
    % Class labels
    labels = ones(1,m*n);
    
    %% (1) Initialization
    centers(:,1) = mean(all_data); % class center
    dists = zeros(1,m*n);
    for p = 1:m*n % points
        diff = all_data(p,:)-centers(:,1)';
        dists(1,p) = sqrt(diff*diff');
    end
    J(1,1) = sum(dists); % distortion
    
    for k = 1:K-1
        %% (2) Choose class with largest distortion
        [~, split_cl] = max(J);
        
        disp(split_cl);

        %% (3) Split the class
        dists_1 = zeros(1,m*n);
        dists_2 = zeros(1,m*n);
        for p = 1:m*n
            if labels(p) == split_cl
                diff = all_data(p,:)-(centers(:,split_cl)'+v);
                dist_1 = sqrt(diff*diff');
                diff = all_data(p,:)-(centers(:,split_cl)'-v);
                dist_2 = sqrt(diff*diff');
                
                if dist_2 < dist_1
                    labels(p) = k+1;
                    dists_2(1,p) = dist_2;
                else
                    dists_1(1,p) = dist_1;
                end
            end
        end
        
        %% (4) Update the centers and distortions
        centers(:,split_cl) = mean(all_data(labels==split_cl,:));
        centers(:,k+1) = mean(all_data(labels==k+1,:));
        
        J(1,split_cl) = sum(dists_1);
        J(1,k+1) = sum(dists_2);
    end
    plot_clusters(all_data, labels, titles(i));
    
end








