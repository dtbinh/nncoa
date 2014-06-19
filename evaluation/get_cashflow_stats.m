function [ stats ] = get_cashflow_stats( input_arrays )

stats = [];

for input_array_cell = input_arrays
    
    input_array = input_array_cell{1,1};
    
    for ticker_index = 1:size(input_array,1)

        combined = combine( input_array, ticker_index, 1 );

        end_profits = combined(end,end,:);

        ministats(ticker_index,1) = min(end_profits);
        ministats(ticker_index,2) = max(end_profits);
        ministats(ticker_index,3) = mean(end_profits);
        ministats(ticker_index,4) = std(end_profits);

    end
    
    stats = [stats;ministats];
    
end




end

