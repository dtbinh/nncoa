function [ all_outputs ] = agent_tester( f, tickers, iters, n_out )

num_tickers = numel(tickers);

ticker_num = 1;

for ticker = tickers
    
    if iscell(ticker) && numel(ticker) == 1
       ticker = ticker{1:1}; 
    end
    
    for i=1:iters
        try
            tic;
            [output{1:n_out}] = f(ticker);
            all_outputs{ticker_num,i} = output;
            t = toc;
            disp(['Round elapsed in ' num2str(t) ' seconds. Estimated time remaining: ' num2str(((num_tickers-ticker_num) * iters + (iters-i))  * t / 60) ' minutes.'])
        catch exception
            continue
        end
    end
    
    save('data.txt','all_outputs');
    
    ticker_num = ticker_num + 1;
end


end

