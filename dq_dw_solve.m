function [ dq_dw_plus, dq_dw_minus ] = dq_dw_solve(w_plus,w_minus,q,r_o,n_out,lambda)

% Set n_out = 0 for the fully recurrent case to avoid r_o being forced.

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(q,'double'));
assert(isreal(q));

assert(isa(r_o,'double'));
assert(isreal(r_o));

assert(isa(n_out,'double'));
assert(isreal(n_out));

assert(isa(lambda,'double'));
assert(isreal(lambda));

assert(isa(w_plus,'double'));
assert(isreal(w_plus));

assert(isa(w_minus,'double'));
assert(isreal(w_minus));

assert ( all(size(q)>=[1,1]));
assert ( all(size(q)<=[1,1000]));

assert ( all(size(r_o)>=[1,1]));
assert ( all(size(r_o)<=[1,1]));

assert ( all(size(n_out)>=[1,1]));
assert ( all(size(n_out)<=[1,1]));

assert ( all(size(lambda)>=[1,1]));
assert ( all(size(lambda)<=[1,1000]));

assert ( all(size(w_plus)>=[1,1]));
assert ( all(size(w_plus)<=[1000,1000]));

assert ( all(size(w_minus)>=[1,1]));
assert ( all(size(w_minus)<=[1000,1000]));

% ====== Main Code ====== %

r = sum(w_plus+w_minus,2)';

r(end+1-n_out:end) = r_o(1);

D = (r + lambda + q * w_minus);

DInv = 1./D;

W = (w_plus - w_minus * diag(q)) * diag(DInv);

IWInv = inv(eye(numel(q))-W);

gamma_plus = zeros([size(W) numel(q)]);
gamma_minus = zeros([size(W) numel(q)]);

dq_dw_plus = zeros([size(W) numel(q)]);
dq_dw_minus = zeros([size(W) numel(q)]);

for i = 1:numel(q)
    gamma_plus(i,:,i) = -1*DInv(i);  % u = i; i.e. row = i;
    gamma_plus(:,i,i) = DInv(i);     % v = i; i.e. column = i;
    gamma_plus(i,i,i) = 0;
    gamma_plus(:,:,i) = diag(q) * gamma_plus(:,:,i);
    
    gamma_minus(i,:,i) = -1 * DInv(i);
    gamma_minus(:,i,i) = -1 * q(i) * DInv(i);
    gamma_minus(i,i,i) = -1 * (1+q(i)) * DInv(i);
    gamma_minus(:,:,i) = diag(q) * gamma_minus(:,:,i);
end

for v = 1:numel(q)
    dq_dw_plus(:,v,:) = squeeze(gamma_plus(:,v,:)) * IWInv;
    dq_dw_minus(:,v,:) = squeeze(gamma_minus(:,v,:)) * IWInv; 
end

end

