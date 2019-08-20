M = 8;
N = 12;

load("A.txt");
load("B.txt");
load("pi.txt");

load("Test.txt");

[T, num_sequences] = size(Test);

%% Forward procedure

probabilities = zeros(1,num_sequences);

for O = 1:num_sequences
    
    % Initialization
    alpha = zeros(1,N);
    for i = 1:N
        alpha(i) = pi(i)*B(Test(1,O),i);
    end
    
    % Induction
    for t = 1:T-1
        for j = 1:N
            sum = 0;
            for i = 1:N
                sum = sum + alpha(i)*A(i,j);
            end
            alpha(j) = sum*B(Test(t+1,O),j);
        end
    end 
    
    % Termination
    for i = 1:N
        probabilities(O) = probabilities(O) + alpha(i);
    end
end

classification = zeros(1,num_sequences);
for i = 1:num_sequences
   if log(probabilities) > -115
       classification(i) = 1;
   else
       classification(i) = 2;
   end
end

