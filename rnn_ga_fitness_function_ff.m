function [ fitness ] = rnn_ga_fitness_function_ff( weight_vector, train_inputwindow, train_ret, threshholds, n_in, n_hidden )

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(weight_vector,'double'));
assert(isreal(weight_vector));

assert(isa(train_inputwindow,'double'));
assert(isreal(train_inputwindow));

assert(isa(train_ret,'double'));
assert(isreal(train_ret));

assert(isa(threshholds,'double'));
assert(isreal(threshholds));

assert(isa(n_in,'double'));
assert(isreal(n_in));

assert(isa(n_hidden,'double'));
assert(isreal(n_hidden));

assert ( all(size(threshholds)==[1,2]));

assert ( all(size(n_in)==[1,1]));
assert ( all(size(n_hidden)==[1,1]));

assert ( all(size(weight_vector)>=[1,1]));
assert ( all(size(weight_vector)<=[1,Inf]));

assert ( all(size(train_inputwindow)>=[1,1]));
assert ( all(size(train_inputwindow)<=[Inf,Inf]));

assert ( all(size(train_ret)>=[1,1]));
assert ( all(size(train_ret)<=[Inf,1]));

% ====== Main Code ====== %

[ w_plus, w_minus ] = convert2wmatrix_ff( weight_vector, n_in, n_hidden );

training_predictions = test_rnn_ff(2, train_inputwindow, w_plus, w_minus, 0.1);
bipolar_training_predictions = training_predictions(:,1) - training_predictions(:,2);

profit = profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,threshholds),1);

fitness = profit(end);

end