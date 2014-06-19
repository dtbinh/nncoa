function [ q ] = q_solve_ff( w_plus, w_minus, r_o, Lambda, lambda, n_in, n_hidden )

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(Lambda,'double'));
assert(isreal(Lambda));

assert(isa(lambda,'double'));
assert(isreal(lambda));

assert(isa(r_o,'double'));
assert(isreal(r_o));

assert(isa(n_in,'double'));
assert(isreal(n_in));

assert(isa(n_hidden,'double'));
assert(isreal(n_hidden));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert ( all(size(r_o)==[1,1]));

assert ( all(size(n_hidden)>=[1,1]));
assert ( all(size(n_hidden)<=[1,1]));

assert ( all(size(n_in)>=[1,1]));
assert ( all(size(n_in)<=[1,1]));

assert ( all(size(Lambda)>=[1,1]));
assert ( all(size(Lambda)<=[1,1000]));

assert ( all(size(lambda)>=[1,1]));
assert ( all(size(lambda)<=[1,1000]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));

% ====== Main Code ====== %


w_plus_ih = w_plus(1:n_in,(n_in+1):(n_in+n_hidden));
w_minus_ih = w_minus(1:n_in,(n_in+1):(n_in+n_hidden));

w_plus_ho = w_plus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end);
w_minus_ho = w_minus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end);

r_i = sum(w_plus_ih + w_minus_ih,2)';
r_h = sum(w_plus_ho + w_minus_ho,2)';

N_i = Lambda;
D_i = lambda + r_i;

q_i = min(1,N_i ./ D_i);


N_h = q_i * w_plus_ih;
D_h = r_h + q_i * w_minus_ih;

q_h = min(1,N_h ./ D_h);


N_o = q_h * w_plus_ho;
D_o = r_o(1) + q_h * w_minus_ho;

q_o = min(1, N_o ./ D_o);

q = [ q_i, q_h, q_o ];

end

