x = [];

%[ w_plus_ih, w_minus_ih,  w_plus_ho, w_minus_ho, r_o ] = rnn_init_ff( 20, 30, 2 );

ns = 40;

[ w_plus, w_minus ] = w_init( ns );

dbstop if error

minerr = Inf;

for i=1:100

%     disp(num2str(i));
%     
%     m = randsample(990,30);
%     
%     input_minibatch = input(m,:);
%     target_minibatch = target(m,:);
    
    x(i) = mae_rnn_mex(inputdata, targetdata, w_plus, w_minus);

    if (x(i) < minerr)
        w_plus_best = w_plus;
        w_minus_best = w_minus;
        minerr = x(i);
    end
        
    [ w_plus, w_minus ] = train_rnn_mex(inputdata, targetdata, 0.01, w_plus, w_minus);
    
end

plot(x);