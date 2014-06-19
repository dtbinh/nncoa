function [ weight_vector ] = convert2wvector_ff( w_plus, w_minus, n_in, n_hidden )

    w_plus_ih = w_plus(1:n_in,(n_in+1):(n_in+n_hidden));
    w_minus_ih = w_minus(1:n_in,(n_in+1):(n_in+n_hidden));

    w_plus_ho = w_plus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end);
    w_minus_ho = w_minus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end);

    weight_vector = [reshape(w_plus_ih, 1, numel(w_plus_ih)), reshape(w_plus_ho, 1, numel(w_plus_ho)),reshape(w_minus_ih, 1, numel(w_minus_ih)), reshape(w_minus_ho, 1, numel(w_minus_ho))];

end