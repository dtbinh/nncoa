x = [];

%[ w_plus_ih, w_minus_ih,  w_plus_ho, w_minus_ho, r_o ] = rnn_init_ff( 20, 30, 2 );

ns = 30;

r_o = 0.1;

[ w_plus, w_minus ] = w_init( ns );

dbstop if error

minerr = Inf;

for i=1:50

%     disp(num2str(i));
%     
%     m = randsample(990,30);
%     
%     input_minibatch = input(m,:);
%     target_minibatch = target(m,:);
    
    x(i) = mae_rnn_ff_mex(inputdata, targetdata, w_plus, w_minus, r_o);

    if (x(i) < minerr)
        w_plus_best = w_plus;
        w_minus_best = w_minus;
        minerr = x(i);
    end
        
    [ w_plus, w_minus ] = train_rnn_ff_mex(inputdata, targetdata, 0.001, w_plus, w_minus, r_o);
    
end

plot(x);