function [ q ] = q_solve( w_plus, w_minus, Lambda, lambda )

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(Lambda,'double'));
assert(isreal(Lambda));

assert(isa(lambda,'double'));
assert(isreal(lambda));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert ( all(size(Lambda)>=[1,1]));
assert ( all(size(Lambda)<=[1,1000]));

assert ( all(size(lambda)>=[1,1]));
assert ( all(size(lambda)<=[1,1000]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));

% ====== Main Code ====== %

r = sum(w_plus + w_minus,2)';

size_q = numel(Lambda);

qn = ones(1,size_q) ./ 2;
qn1 = zeros(1,size_q);

iter_count = 0;

w_plus(logical(eye(size(w_plus)))) = 0;
w_minus(logical(eye(size(w_minus)))) = 0;

while sum(abs(qn-qn1)) > 0.001 && iter_count < 1000
    qn1 = qn;
    qn = min(1,(Lambda + qn * w_plus) ./ (r + lambda + qn * w_minus));
    iter_count = iter_count + 1;
end

q = qn;

end

