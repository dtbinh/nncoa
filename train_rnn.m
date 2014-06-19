function [ w_plus, w_minus ] = train_rnn(inputdata, targetdata, eta, w_plus, w_minus)

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(inputdata,'double'));
assert(isreal(inputdata));

assert(isa(targetdata,'double'));
assert(isreal(targetdata));

assert(isa(eta,'double'));
assert(isreal(eta));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert ( all(size(inputdata)>=[1,1]));
assert ( all(size(inputdata)<=[Inf,1000]));

assert ( all(size(targetdata)>=[1,1]));
assert ( all(size(targetdata)<=[Inf,1000]));

assert ( all(size(eta)==[1,1]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));

% ====== Main Code ====== %

n_in = size(inputdata,2);
n_out = size(targetdata,2);

n = size(w_plus,1);

assert ( n >= n_in + n_out );

num_training_pairs = size(inputdata,1);

% p = randperm(num_training_pairs);
% inputdata = inputdata(p,:);
% targetdata = targetdata(p,:);

assert(num_training_pairs == size(targetdata,1),'Number of input vectors does not match number of output vectors');

error_weights = zeros(1,1,n);

Lambda = zeros(1,n);
lambda = zeros(1,n);

for k=1:num_training_pairs
    
    Lambda(1:n_in) = max(inputdata(k,:),0);
    lambda(1:n_in) = abs(min(inputdata(k,:),0));

    q = q_solve( w_plus, w_minus, Lambda, lambda );
    
    error_weights(n_in+1:n_in+n_out) = q(n_in+1:n_in+n_out) - targetdata(k,:);

    [ dq_dw_plus, dq_dw_minus ] = dq_dw_solve(w_plus,w_minus,q,0,0,lambda);

    w_plus = max(0,w_plus - eta .* sum(bsxfun(@times,dq_dw_plus,error_weights),3));      % Calculating more derivatives than necessary.
    w_minus = max(0,w_minus - eta .* sum(bsxfun(@times,dq_dw_minus,error_weights),3));   % This should be optimized out at some point.
    
end

end

