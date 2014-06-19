function [ w_plus, w_minus ] = w_init( n )

w_plus = rand(n) .* 0.2;
w_minus = rand(n) .* 0.2;

end

