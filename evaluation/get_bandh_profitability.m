function [ profits ] = get_bandh_profitability( tickers, days )

ticker_index = 1;
for ticker = tickers
   test_data = fetch(yahoo,ticker,'Adj Close',  '01/01/13', '12/31/13'); 
   test_data = flipud(test_data(:,2));
   profits(ticker_index) = test_data(end)/test_data(end+1-days);
   ticker_index = ticker_index + 1;
end


end

