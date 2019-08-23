function [policy, iterations] = WalkPolicyIteration(s)
    % reward matrix
    rew = [ 0  0  0  0;
            0  1 -1 -1;
            0 -1 -1 -1;
            0  0  0  0;
           -1 -1  0  1;
            0  0  0  0;
            0  0  0  0;
           -1  1  0  0;
           -1 -1  0 -1;
            0  0  0  0;
            0  0  0  0;
           -1  1  0 -1;
            0  0  0  0;
            0  0 -1  1;
            0 -1 -1  1;
            0  0  0  0];

    % state transition matrix   
    state_trans = [ 2  4  5 13;
                    1  3  6 14;
                    4  2  7 15;
                    3  1  8 16;
                    6  8  1  9;
                    5  7  2 10;
                    8  6  3 11;
                    7  5  4 12;
                   10 12 13  5;
                    9 11 14  6;
                   12 10 15  7;
                   11  9 16  8;
                   14 16  9  1;
                   13 15 10  2;
                   16 14 11  3;
                   15 13 12  4];
    
    % random initial policy
    policy = ceil(rand(16,1)*4);
    
    old_policy = policy;
    iterations = 0;
    num_states = 16;
    num_actions = 4;
    gamma = 0.6;
    converged = false;
    
    % repeat until convergence
    while ~ converged
        iterations = iterations + 1
        
        % 16 equations with 16 unknowns
        a = zeros(16,16);
        b = zeros(16, 1);
        
        % Bellman equation
        for state = 1:num_states
            a(state, state_trans(state, policy(state))) = -gamma;
            b(state) = rew(state, policy(state));
        end
        value = a\b;
        
        % greedy updates
        for state = 1:num_states
            max_value = -inf;
            for action = 1:num_actions
                test_value = rew(state, action) + ...
                    gamma*value(state_trans(state,action));
                if test_value > max_value
                    policy(state) = action;
                    max_value = test_value;
                end
            end
        end
        
        if all(old_policy == policy)
            converged = true;
        end
        
        old_policy = policy;
    end
    
    states = zeros(1,16);
    states(1) = s;
    for i = 2:num_states
        states(i) = state_trans(states(i-1), policy(states(i-1)));
    end
    
    figure
    walkshow(states);
end






