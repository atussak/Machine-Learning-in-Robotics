function plot_clusters(points, labels, Title)
    k = 7;
    colors = ['b', 'k', 'r', 'g', 'm', 'y', 'c'];

    figure  
    hold on
   
    for c = 1:k
        plot3(points(labels==c,1), points(labels==c,2), ...
            points(labels==c,3), '.', ...
            'color', colors(c), ...
            'MarkerSize', 12);
    end
    
    title(Title);

end