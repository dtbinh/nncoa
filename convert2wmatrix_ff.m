function [ w_plus, w_minus ] = convert2wmatrix_ff( weight_vector, n_in, n_hidden )
    
    % ====== Assertions for MATLAB Coder Compatibility ====== %

    assert(isa(weight_vector,'double'));
    assert(isreal(weight_vector));

    assert(isa(n_in,'double'));
    assert(isreal(n_in));

    assert(isa(n_hidden,'double'));
    assert(isreal(n_hidden));

    assert ( all(size(n_in)==[1,1]));
    
    assert ( all(size(n_hidden)==[1,1]));

    assert ( all(size(weight_vector)>=[1,1]));
    assert ( all(size(weight_vector)<=[1,Inf]));

    % ====== Main Code ====== %

    w_plus = zeros(n_in + n_hidden + 2);
    w_minus = zeros(n_in + n_hidden + 2);

    ih_size = n_in * n_hidden;
    ho_size = n_hidden * 2;

    w_plus_ih = reshape(weight_vector(1:ih_size),n_in,n_hidden);
    w_plus_ho = reshape(weight_vector((ih_size+1):(ih_size+ho_size)),n_hidden,2);
    
    w_minus_ih = reshape(weight_vector((ih_size+ho_size+1):(ih_size+ho_size+ih_size)),n_in,n_hidden);
    w_minus_ho = reshape(weight_vector((ih_size+ho_size+ih_size+1):end),n_hidden,2);
    
    w_plus(1:n_in,(n_in+1):(n_in+n_hidden)) = w_plus_ih;
    w_minus(1:n_in,(n_in+1):(n_in+n_hidden)) = w_minus_ih;

    w_plus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end) = w_plus_ho;
    w_minus((n_in+1):(n_in+n_hidden),(n_in+n_hidden+1):end) = w_minus_ho;

end

