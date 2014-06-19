function [ y ] = rectify( x )

y = [max(0,x);abs(min(0,x))];

end

