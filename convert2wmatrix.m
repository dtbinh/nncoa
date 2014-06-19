function [ w_plus, w_minus ] = convert2wmatrix( weight_vector )

m_numel = numel(weight_vector)/2;

side_length = sqrt(m_numel);

w_plus = reshape(weight_vector(1:m_numel),side_length,side_length);
w_minus = reshape(weight_vector(m_numel+1:end),side_length,side_length);

end

