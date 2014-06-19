function [ outputdata ] = test_rnn(n_out, inputdata, w_plus, w_minus)

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(inputdata,'double'));
assert(isreal(inputdata));

assert(isa(n_out,'double'));
assert(isreal(n_out));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert ( all(size(inputdata)>=[1,1]));
assert ( all(size(inputdata)<=[Inf,1000]));

assert ( all(size(n_out)==[1,1]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));

% ====== Main Code ====== %

n_in = size(inputdata,2);

n = size(w_plus,1);

outputdata = zeros(size(inputdata,1),n_out);

for k=1:size(outputdata,1)
    Lambda = zeros(1,n);
    lambda = zeros(1,n);

    Lambda(1:n_in) = max(inputdata(k,:),0);
    lambda(1:n_in) = abs(min(inputdata(k,:),0));
    
    result = q_solve( w_plus, w_minus, Lambda, lambda );
    
    outputdata(k,:) =  result(n_in+1:n_in+n_out);
end

end

