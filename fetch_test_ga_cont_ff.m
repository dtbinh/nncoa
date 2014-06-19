function [ cashflow, training_cashflow, decisions, w_plus_best, w_minus_best, threshholds ] = fetch_test_ga_cont_ff( tickers, tr_start_date, tr_end_date, te_start_date, te_end_date )

    % Output is the predictions for the future price of the first ticker in
    % tickers.

    warning('off','all');

    num_tickers = numel(tickers);
    
    assert(num_tickers < 5);
    
    % TRAIN ++++++++++++++++++++++++++++++++++++++++++++++++
    
    ticker_raw = fetch(yahoo,tickers{1},'Adj Close', tr_start_date,tr_end_date);
    
    training_fts = fints(ticker_raw(:,1),ticker_raw(:,2),regexprep(tickers{1},'\^',''));
    
    for ticker = tickers(2:end)
        
        ticker_raw = fetch(yahoo,ticker,'Adj Close', tr_start_date,tr_end_date);
        
        ticker_ts = fints(ticker_raw(:,1),ticker_raw(:,2),regexprep(ticker,'\^',''));
        
        training_fts = merge(training_fts,ticker_ts,'DateSetMethod','intersection');
        
    end
    
    synchronized_training_data = fts2mat(training_fts);
    
    train_data_stationary = zeros([size(synchronized_training_data,1)-30,size(synchronized_training_data,2)]);
    
    for i = 1:numel(tickers)
       
        % This could be vectorized
        [~,train_data_ma] = movavg(synchronized_training_data(:,i),1,30);
        train_data_stationary(:,i) = (synchronized_training_data(31:end,i) - train_data_ma(30:end-1)) .* 0.1;
        
    end
    
    % TEST  +++++++++++++++++++++++++++++++++++++++++++++++++
    
    ticker_raw = fetch(yahoo,tickers{1},'Adj Close', te_start_date, te_end_date);
    
    test_fts = fints(ticker_raw(:,1),ticker_raw(:,2),regexprep(tickers{1},'\^',''));
    
    for ticker = tickers(2:end)
        
        ticker_raw = fetch(yahoo,ticker,'Adj Close', te_start_date, te_end_date);
        
        ticker_ts = fints(ticker_raw(:,1),ticker_raw(:,2),regexprep(ticker,'\^',''));
        
        test_fts = merge(test_fts,ticker_ts,'DateSetMethod','intersection');
        
    end
    
    synchronized_test_data = fts2mat(test_fts);
    
    test_data_stationary = zeros([size(synchronized_test_data,1)-30,size(synchronized_test_data,2)]);
    
    for i = 1:numel(tickers)
       
        % This could be vectorized
        [~,test_data_ma] = movavg(synchronized_test_data(:,i),1,30);
        test_data_stationary(:,i) = (synchronized_test_data(31:end,i) - test_data_ma(30:end-1)) .* 0.1;
        
    end
    
    % +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    [ train_inputwindow, train_target ] = createtrainingdata( train_data_stationary' , 10 );
    [ test_inputwindow, test_target ] = createtrainingdata( test_data_stationary' , 10 );
    
    train_ret = tick2ret(synchronized_training_data(:,1));
    test_ret = tick2ret(synchronized_test_data(:,1));
    
    options = gaoptimset('UseParallel','always');
    
    n_in = 20*num_tickers;
    n_hidden = ceil(500/(2+20*num_tickers));
    
    n_total = (n_in+2) * n_hidden * 2;
    
    weight_vector = ga(@(weight_vector)(-1 * rnn_ga_fitness_function_ff_mex( weight_vector, train_inputwindow, train_ret, [0,0], n_in, n_hidden)),n_total,[],[],[],[],zeros(1,n_total),[],[],options);
    
    [ w_plus_best, w_minus_best ] = convert2wmatrix_ff( weight_vector, n_in, n_hidden );
    
    training_predictions = test_rnn_ff(2, train_inputwindow, w_plus_best, w_minus_best, 0.1);
    bipolar_training_predictions = training_predictions(:,1) - training_predictions(:,2);
    
    threshholds = [0 0];
    
    training_cashflow = profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,threshholds),1);
    
    period_length = 10;
    
    num_periods = ceil(size(test_inputwindow,1)/period_length);
        
    for i = 1:num_periods

        period_start_day = (period_length*(i-1))+1;
        period_end_day = min((period_length*(i-1))+period_length,size(test_inputwindow,1));
        
        test_predictions(period_start_day:period_end_day,:) = test_rnn_ff(2, test_inputwindow(period_start_day:period_end_day,:), w_plus_best, w_minus_best, 0.1);
        
        options2 = gaoptimset('InitialPopulation',weight_vector,'Generations',10,'UseParallel','always');
        
        if i == 1 
            continue;
        end
        
        weight_vector = ga(@(weight_vector)(-1 * rnn_ga_fitness_function_ff_mex( weight_vector, test_inputwindow(1:(period_start_day-1),:), test_ret, [0,0], n_in, n_hidden)),n_total,[],[],[],[],zeros(1,n_total),[],[],options2);
        [ w_plus_best, w_minus_best ] = convert2wmatrix_ff( weight_vector, n_in, n_hidden );
    end
        
    bipolar_predictions = test_predictions(:,1) - test_predictions(:,2);
    
    decisions = decision_maker(bipolar_predictions,threshholds);
    
    cashflow = profit_calc( test_ret(end+1-numel(decisions):end), decisions, 1 );

    figure
    subplot(2,1,1);
    plot(1:numel(cashflow),cashflow,1:numel(cashflow),profit_calc( test_ret(end+1-numel(decisions):end), ones(size(decisions)), 1 ));
    xlim([0,numel(cashflow)]);
    
    subplot(2,1,2);
    stairs(decisions);
    xlim([0,numel(cashflow)]);
    ylim([-1.5,1.5]);
end

