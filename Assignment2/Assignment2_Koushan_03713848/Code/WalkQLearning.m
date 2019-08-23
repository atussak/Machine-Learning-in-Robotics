function WalkQLearning(s)
    g = 0.5;
    e = 0.4;
    a = 1;
    
    Q = zeros(16,4);
    first_s = s;
    
    num_states = 16;
    iters = 300;
    
    for i = 1:iters
        % choose action from s using e-greedy policy
        % based on Q(state, action)
        action = 0;
        if rand <= e
            action = ceil(4*rand);
        else
            [~, action] = max(Q(s,:));
        end
        
        [next_s, r] = SimulateRobot(s, action);
        Q(s, action) = Q(s, action) + a*(r+g*max(Q(next_s,:))-Q(s, action));
        s = next_s;
    end
    [~, policy] = max(Q');
    states = zeros(1,16);
    states(1) = first_s;
    for i = 2:num_states
        [states(i), ~] = SimulateRobot(states(i-1), policy(states(i-1)));
    end
    
    walkshow(states);
end






