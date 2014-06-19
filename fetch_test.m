function [ cashflow, decisions, w_plus_best, w_minus_best, threshholds ] = fetch_test( ticker, tr_start_date, tr_end_date, te_start_date, te_end_date )

    train_data = fetch(yahoo,ticker,'Adj Close', tr_start_date,tr_end_date);
    test_data = fetch(yahoo,ticker,'Adj Close', te_start_date,te_end_date);

    train_data = flip(train_data,1);
    test_data = flip(test_data,1);
    
    train_ret = tick2ret(train_data(:,2));
    test_ret = tick2ret(test_data(:,2));
    
    train_ret_mod = train_ret .* 10;
    test_ret_mod = test_ret .* 10;
    
    [ train_inputwindow, train_target ] = createtrainingdata( train_ret_mod' , 10 );
    [ test_inputwindow, test_target ] = createtrainingdata( test_ret_mod' , 10 );
    
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
    
    training_predictions = test_rnn_ff_mex(2, train_inputwindow, w_plus_best, w_minus_best, 0.1);
    bipolar_training_predictions = training_predictions(:,1) - training_predictions(:,2);
    
    %threshholds = ga(@(x)(-1*sharpe(tick2ret(profit_calc(train_ret(11:end),decision_maker(bipolar_training_predictions,x),1)))),2);
    threshholds = [0 0];
    
    test_predictions = test_rnn_ff_mex(2, test_inputwindow, w_plus_best, w_minus_best, 0.1);
    
    bipolar_predictions = test_predictions(:,1) - test_predictions(:,2);
    
    decisions = decision_maker(bipolar_predictions,threshholds);
    
    cashflow = profit_calc( test_ret(11:end), decisions, 1 );

    plot(1:numel(cashflow),cashflow,1:numel(cashflow),profit_calc( test_ret(11:end), ones(size(decisions)), 1 ));
    
end

