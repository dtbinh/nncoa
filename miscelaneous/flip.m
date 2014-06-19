function [ y ] = flip( x, d )

switch d
    case 1
        y = flipud(x);
    case 2
        y = fliplr(x);
end


end

