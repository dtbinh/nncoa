function [ w_plus, w_minus ] = convert2wmatrix_ff_xor( weights )
    % ====== Main Code ====== %

    n_in = 2;
    n_hidden = 3;
    n_out = 1;
    
    w_plus = zeros(n_in+n_hidden+n_out);
    w_minus = zeros(n_in+n_hidden+n_out);
    
    
    w_plus_ih = reshape(weights(1:6),2,3);
    w_minus_ih = reshape(weights(7:12),2,3);
    w_plus_ho = reshape(weights(13:15),3,1);
    w_minus_ho = reshape(weights(16:18),3,1);
    
    w_plus(1:n_in,(n_in+1):(n_in+n_hidden)) = w_plus_ih;
    w_minus(1:n_in,(n_in+1):(n_in+n_hidden)) = w_minus_ih;

    w_plus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end) = w_plus_ho;
    w_minus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end) = w_minus_ho;

end

