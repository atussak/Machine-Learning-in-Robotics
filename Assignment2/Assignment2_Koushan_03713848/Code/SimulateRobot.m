function [newstate, reward] = SimulateRobot(state, action)
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

   reward = rew(state, action);
   newstate = state_trans(state, action);
end