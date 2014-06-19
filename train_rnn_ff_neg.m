function [ w_plus, w_minus ] = train_rnn_ff_neg(inputdata, targetdata, eta, w_plus, w_minus, r_o)

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(inputdata,'double'));
assert(isreal(inputdata));

assert(isa(targetdata,'double'));
assert(isreal(targetdata));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert(isa(r_o,'double'));
assert(isreal(r_o));

assert(isa(eta,'double'));
assert(isreal(eta));

assert ( all(size(eta)==[1,1]));

assert ( all(size(inputdata)>=[1,1]));
assert ( all(size(inputdata)<=[Inf,1000]));

assert ( all(size(targetdata)>=[1,1]));
assert ( all(size(targetdata)<=[Inf,1000]));

assert ( all(size(r_o)>=[1,1]));
assert ( all(size(r_o)<=[1,1000]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));


% ====== Main Code ====== %

n_in = size(inputdata,2);
n_out = size(targetdata,2);

n = size(w_plus,1);

n_hidden = n - (n_in + n_out);

assert ( n >= n_in + n_out );

mask = zeros(n);
mask(1:n_in,(n_in+1):(n_in+n_hidden)) = 1;
mask((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end) = 1;
mask = ~mask;

w_plus(mask) = 0;
w_minus(mask) = 0;

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

    q = q_solve_ff( w_plus, w_minus, r_o, Lambda(1:n_in), lambda(1:n_in), n_in, n_hidden );
    
    error_weights(end+1-n_out:end) = q(end+1-n_out:end) - targetdata(k,:);

    [ dq_dw_plus, dq_dw_minus ] = dq_dw_solve(w_plus,w_minus,q, r_o, n_out,lambda);

%     w_plus = max(0,w_plus - nu .* sum(bsxfun(@times,dq_dw_plus,error_weights),3));      % Calculating more derivatives than necessary.
%     w_minus = max(0,w_minus - nu .* sum(bsxfun(@times,dq_dw_minus,error_weights),3));   % This should be optimized out at some point.
    
    w_plus = w_plus - eta .* sum(bsxfun(@times,dq_dw_plus,error_weights),3);      % Calculating more derivatives than necessary.
    w_minus = w_minus - eta .* sum(bsxfun(@times,dq_dw_minus,error_weights),3);   % This should be optimized out at some point.
    
    w_plus(mask) = 0;
    w_minus(mask) = 0;
    
%     row_total = sum(w_plus + w_minus,2);
%     row_total(row_total == 0) = 1;
%     w_plus = bsxfun(@rdivide,w_plus,row_total);
%     w_minus = bsxfun(@rdivide,w_minus,row_total);
    
end

end

