function Exercise3_kmeans(gesture_l, ...
    gesture_o, gesture_x, init_cluster_l, init_cluster_o, ...,
    init_cluster_x, k)

titles = ["l-gesture", "o-gesture", "x-gesture"];

%% (1) Initialization
gestures = {gesture_l, gesture_o, gesture_x};
init_clusters = {init_cluster_l, init_cluster_o, init_cluster_x};
clusters = init_clusters;
num_gests = size(gestures);

for i = 1:num_gests(2)
    [m,n,o] = size(gestures{i});
    gesture = reshape(gestures{i},[m*n,o]);
    centers = clusters{i};
    J = inf;
    decrement = inf;
    
    while decrement >= 10e-6

        %% (2) E-step
        labels = zeros(1, m*n);
        min_dists = zeros(1, m*n);

        for p = 1:m*n % points
            dists = zeros(1,k);
            for c = 1:k % existing cluster centers
                diff = gesture(p,:)-centers(c,:);
                dists(1,c) = sqrt(diff*diff');
            end
            % Find the closest class for the point and label it
            [min_dists(p), labels(p)] = min(dists);
        end

        %% (3) M-step
        for c = 1:k
            centers(c,:) = mean(gesture(labels==c,:));
        end

        %% (4) Distortion
        J_new = sum(min_dists);
    
        %% (5) Convergence
        decrement = J - J_new;
        J = J_new;
        
    end
    
    plot_clusters(gesture, labels, titles(i));
end