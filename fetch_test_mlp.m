function [ cashflow, training_cashflow, decisions, threshholds ] = fetch_test_mlp( ticker, tr_start_date, tr_end_date, te_start_date, te_end_date )

    train_data = fetch(yahoo,ticker,'Adj Close', tr_start_date,tr_end_date);
    test_data = fetch(yahoo,ticker,'Adj Close', te_start_date,te_end_date);

    train_data = flip(train_data,1);
    test_data = flip(test_data,1);
    
    [~,train_data_ma] = movavg(train_data(:,2),1,30);
    [~,test_data_ma] = movavg(test_data(:,2),1,30);
    
    train_data_stationary = (train_data(31:end,2) - train_data_ma(31:end)) .* 0.1;
    test_data_stationary = (test_data(31:end,2) - test_data_ma(31:end)) .* 0.1;
    
    windowsize = 10;
    
    [ train_inputwindow, train_target ] = createtrainingdata( train_data_stationary' , windowsize );
    [ test_inputwindow, test_target ] = createtrainingdata( test_data_stationary' , windowsize );
    
    train_inputwindow = train_inputwindow(:,1:windowsize) - train_inputwindow(:,(windowsize+1):end);
    test_inputwindow = test_inputwindow(:,1:windowsize) - test_inputwindow(:,(windowsize+1):end);
    
    train_target = train_target(:,1) - train_target(:,2);
    test_target = test_target(:,1) - test_target(:,2);
    
    train_ret = tick2ret(train_data(:,2));
    test_ret = tick2ret(test_data(:,2));
    
    % ==============================================
    
    net = feedforwardnet(50);
    
    net = train(net,train_inputwindow',train_target','useParallel','yes');
    
    % ==============================================
    
    bipolar_training_predictions = net(train_inputwindow')';
    
    %threshholds = ga(@(x)(-1*sharpe(tick2ret(profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,x),1)))),2);
    threshholds = [0 0];
    
    training_cashflow = profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,threshholds),1);
    
    bipolar_predictions = net(test_inputwindow')';
    
    decisions = decision_maker(bipolar_predictions,threshholds);
    
    cashflow = profit_calc( test_ret(end+1-numel(decisions):end), decisions, 1 );

    plot(1:numel(cashflow),cashflow,1:numel(cashflow),profit_calc( test_ret(end+1-numel(decisions):end), ones(size(decisions)), 1 ));
    
end

