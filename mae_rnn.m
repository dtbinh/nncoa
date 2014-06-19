function [ error ] = mae_rnn(inputdata, targetdata, w_plus, w_minus)

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(inputdata,'double'));
assert(isreal(inputdata));

assert(isa(targetdata,'double'));
assert(isreal(targetdata));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert ( all(size(inputdata)>=[1,1]));
assert ( all(size(inputdata)<=[Inf,1000]));

assert ( all(size(targetdata)>=[1,1]));
assert ( all(size(targetdata)<=[Inf,1000]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));

% ====== Main Code ====== %

error = 0;

n_in = size(inputdata,2);

n_out = size(targetdata,2);

n = size(w_plus,1);

for k=1:size(inputdata,1)
    Lambda = zeros(1,n);
    lambda = zeros(1,n);

    Lambda(1:n_in) = max(inputdata(k,:),0);
    lambda(1:n_in) = abs(min(inputdata(k,:),0));
    
    result = q_solve( w_plus, w_minus, Lambda, lambda );
    
    error = error + sum(abs(result(n_in+1:n_in+n_out) - targetdata(k,:)));  
    
end

error = error / size(inputdata,1);

end

