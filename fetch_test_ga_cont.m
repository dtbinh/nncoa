function [ cashflow, training_cashflow, decisions, w_plus_best, w_minus_best, threshholds ] = fetch_test_ga_cont( ticker, tr_start_date, tr_end_date, te_start_date, te_end_date )

    warning('off','all');

    train_data = fetch(yahoo,ticker,'Adj Close', tr_start_date,tr_end_date);
    test_data = fetch(yahoo,ticker,'Adj Close', te_start_date,te_end_date);
    
    train_data = flip(train_data,1);
    test_data = flip(test_data,1);
    
    [~,train_data_ma] = movavg(train_data(:,2),1,30);
    [~,test_data_ma] = movavg(test_data(:,2),1,30);
    
    train_data_stationary = (train_data(31:end,2) - train_data_ma(30:end-1)) .* 0.1;
    test_data_stationary = (test_data(31:end,2) - test_data_ma(30:end-1)) .* 0.1;
    
    [ train_inputwindow, train_target ] = createtrainingdata( train_data_stationary' , 10 );
    [ test_inputwindow, test_target ] = createtrainingdata( test_data_stationary' , 10 );
    
    train_ret = tick2ret(train_data(:,2));
    test_ret = tick2ret(test_data(:,2));
    
    % need to train UNTIL convergence
    
    options = gaoptimset('UseParallel','always');
    
    weight_vector = ga(@(weight_vector)(-1 * rnn_ga_fitness_function_mex( weight_vector, train_inputwindow, train_ret, [0,0])),968,[],[],[],[],zeros(1,968),[],[],options);
    
    [ w_plus_best, w_minus_best ] = convert2wmatrix( weight_vector );
    
    training_predictions = test_rnn(2, train_inputwindow, w_plus_best, w_minus_best);
    bipolar_training_predictions = training_predictions(:,1) - training_predictions(:,2);
    
    
    %threshholds = ga(@(x)(-1*sharpe(tick2ret(profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,x),1)))),2);
    %options2 = gaoptimset('InitialPopulation',[0,0]);
    
    %threshholds = ga(@(threshholds)(-1 * rnn_ga_fitness_function_mex( weight_vector, train_inputwindow, train_ret, threshholds)),2,[],[],[],[],[0,-Inf],[Inf,0],[],options);
    threshholds = [0 0];
    
    training_cashflow = profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,threshholds),1);
    
    period_length = 10;
    
    num_periods = ceil(size(test_inputwindow,1)/period_length);
        
    for i = 1:num_periods

        period_start_day = (period_length*(i-1))+1;
        period_end_day = min((period_length*(i-1))+period_length,size(test_inputwindow,1));
        
        test_predictions(period_start_day:period_end_day,:) = test_rnn(2, test_inputwindow(period_start_day:period_end_day,:), w_plus_best, w_minus_best);
        
        options2 = gaoptimset('InitialPopulation',weight_vector,'Generations',10,'UseParallel','always');
        
        if i == 1 
            continue;
        end
        
        weight_vector = ga(@(weight_vector)(-1 * rnn_ga_fitness_function_mex( weight_vector, test_inputwindow(1:(period_start_day-1),:), test_ret, [0,0])),968,[],[],[],[],zeros(1,968),[],[],options2);
        [ w_plus_best, w_minus_best ] = convert2wmatrix( weight_vector );
    end
    
    %test_predictions = test_rnn(2, test_inputwindow, w_plus_best, w_minus_best);
    
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

