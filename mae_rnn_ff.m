function [ error ] = mae_rnn_ff(inputdata, targetdata, w_plus, w_minus, r_o)

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

error = 0;

n = size(w_plus,1);

n_in = size(inputdata,2);

n_out = size(targetdata,2);

n_hidden = n - (n_in + n_out);

for k=1:size(inputdata,1)
    Lambda = zeros(1,n);
    lambda = zeros(1,n);

    Lambda(1:n_in) = max(inputdata(k,:),0);
    lambda(1:n_in) = abs(min(inputdata(k,:),0));
    
    q = q_solve_ff( w_plus, w_minus, r_o, Lambda(1:n_in), lambda(1:n_in), n_in, n_hidden );
    
    error = error + sum(abs(q(end+1-n_out:end) - targetdata(k,:)));
    
end

error = error / size(inputdata,1);

end

