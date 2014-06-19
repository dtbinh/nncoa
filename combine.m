function [ combined ] = combine( array, ticker_index, output_arg_index )

    combined = [];
    
    num_dims = ndims(array{ticker_index,1}{output_arg_index});

    for i = 1:size(array,2)
        mat = array{ticker_index,i}{output_arg_index};
        
        combined = cat(num_dims+1,combined,mat);
    end

end