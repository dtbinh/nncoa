function [ cashflow, training_cashflow, decisions, w_plus_best, w_minus_best, threshholds ] = fetch_test_ma( ticker, tr_start_date, tr_end_date, te_start_date, te_end_date )

    train_data = fetch(yahoo,ticker,'Adj Close', tr_start_date,tr_end_date);
    test_data = fetch(yahoo,ticker,'Adj Close', te_start_date,te_end_date);
    
    train_data = flip(train_data,1);
    test_data = flip(test_data,1);
    
    [~,train_data_ma] = movavg(train_data(:,2),1,30);
    [~,test_data_ma] = movavg(test_data(:,2),1,30);
    
    train_data_stationary = (train_data(31:end,2) - train_data_ma(31:end)) .* 0.1;
    test_data_stationary = (test_data(31:end,2) - test_data_ma(31:end)) .* 0.1;
    
    [ train_inputwindow, train_target ] = createtrainingdata( train_data_stationary' , 10 );
    [ test_inputwindow, test_target ] = createtrainingdata( test_data_stationary' , 10 );
    
    train_ret = tick2ret(train_data(:,2));
    test_ret = tick2ret(test_data(:,2));
    
    % need to train UNTIL convergence
    
    ns = 50;

    r_o = 0.1;

    [ w_plus, w_minus ] = w_init( ns );

    minerr = Inf;
    
    failed_iterations = 0;
    
    total_iters = 0;
    
    while 1

        disp(['Round ' num2str(total_iters)]);
        
        total_iters = total_iters + 1;
                
        x = mae_rnn_ff_mex(train_inputwindow, train_target, w_plus, w_minus, r_o);

        if (x < minerr * 0.95)
            w_plus_best = w_plus;
            w_minus_best = w_minus;
            minerr = x;
            failed_iterations = 0;
        else
            failed_iterations = failed_iterations + 1;
        end

        if failed_iterations > 6
            break;
        end
        
        [ w_plus, w_minus ] = train_rnn_ff_mex(train_inputwindow, train_target, 0.01, w_plus, w_minus, r_o);

    end
    
    training_predictions = test_rnn_ff(2, train_inputwindow, w_plus_best, w_minus_best, 0.1);
    bipolar_training_predictions = training_predictions(:,1) - training_predictions(:,2);
    
    %threshholds = ga(@(x)(-1*sharpe(tick2ret(profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,x),1)))),2);
    threshholds = [0 0];
    
    training_cashflow = profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,threshholds),1);
    
    test_predictions = test_rnn_ff(2, test_inputwindow, w_plus_best, w_minus_best, 0.1);
    
    bipolar_predictions = test_predictions(:,1) - test_predictions(:,2);
    
    decisions = decision_maker(bipolar_predictions,threshholds);
    
    cashflow = profit_calc( test_ret(end+1-numel(decisions):end), decisions, 1 );

    plot(1:numel(cashflow),cashflow,1:numel(cashflow),profit_calc( test_ret(end+1-numel(decisions):end), ones(size(decisions)), 1 ));
    
end

