load('Data.mat');

k = 5; % k-fold cross validation
n = size(Input, 2);
P = 6;

% vectors of errors with index corresponding to the order of polynomial p
position_errors = zeros(1,6);
orientation_errors = zeros(1,6);

min_pos_err = intmax;
min_orient_err = intmax;

% Optimal model complexities (pol order)
p1 = 0;
p2 = 0;

for p = 1:P

    sum_pos_err = 0;
    sum_orient_err = 0;
    
    for K = 1:k
        % partition no. K test indexes: 1+(K-1)*n/k ---> K*n/k
        % the rest will be used for training
        test_idxs = 1+(K-1)*n/k:K*n/k;
        all_idxs = 1:n;
        train_idxs = setdiff(all_idxs, test_idxs);
        
        % Input values for training
        v_train = Input(1, train_idxs);
        w_train = Input(2, train_idxs);
        
        % Output values for training
        x_train = Output(1, train_idxs);
        y_train = Output(2, train_idxs);
        theta_train = Output(3, train_idxs);
        
        % Input values for testing
        v_test = Input(1, test_idxs);
        w_test = Input(2, test_idxs);
   
        % Output values for testing
        x_test = Output(1, test_idxs);
        y_test = Output(2, test_idxs);
        theta_test = Output(3, test_idxs);
        
        % size of partition
        m = size(v_train', 1);
        % first row must be ones (corresponding to the constant in
        %                                       the linear equation)
        X = ones(m, 1); 
        
        % f(x) = Xa
        for i = 1:p
            X = [X (v_train').^i (w_train').^i (v_train'.*w_train').^i];
        end
        
        % Learn parameters
        a1 = inv(X'*X)*X'*x_train';
        a2 = inv(X'*X)*X'*y_train';
        a3 = inv(X'*X)*X'*theta_train';
        
        % Make X for testing
        m = size(v_test', 1);
        X = ones(m, 1); 
        for i = 1:p
            X = [X (v_test').^i (w_test').^i (v_test'.*w_test').^i];
        end
        
        % Predict output values
        x_pred = X*a1;
        y_pred = X*a2;
        theta_pred = X*a3;
        
        % Position and orientation error
        test_size = n/k;
        pos_err = 0;
        orient_err = 0;
        for i = 1:test_size
            pos_err = pos_err + ...
                sqrt((x_test(i)-x_pred(i))^2 + (y_test(i)-y_pred(i))^2);
            orient_err = orient_err + ...
                sqrt((theta_test(i)-theta_pred(i))^2);
        end
        pos_err = pos_err/test_size;
        orient_err = orient_err/test_size;
        
        % For including all k rounds when comparing model complexities
        sum_pos_err = sum_pos_err + pos_err;
        sum_orient_err = sum_orient_err + orient_err;
        
    end
    
    if(sum_pos_err < min_pos_err)
        min_pos_err = sum_pos_err;
        p1 = p;
    end

    if(sum_orient_err < min_orient_err)
        min_orient_err = sum_orient_err;
        p2 = p;
    end
  
end

% Final parameter estimation based on optimal polynomial orders p1 and p2
v = Input(1,:);
w = Input(2,:);
x = Output(1,:);
y = Output(2,:);
theta = Output(3,:);

X = ones(n, 1);
for i = 1:p1
    X = [X (v').^i (w').^i (v'.*w').^i];
end
par1 = inv(X'*X)*X'*x';
par2 = inv(X'*X)*X'*y';

X = ones(n, 1);
for i = 1:p2
    X = [X (v').^i (w').^i (v'.*w').^i];
end
par3 = inv(X'*X)*X'*theta';


% Save learned parameters
par = {par1, par2, par3};
save('params', 'par');
