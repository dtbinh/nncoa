function [ weight_vector ] = convert2wvector( w_plus, w_minus )

    weight_vector = [reshape(w_plus, 1, numel(w_plus)),reshape(w_minus, 1, numel(w_minus))];

end