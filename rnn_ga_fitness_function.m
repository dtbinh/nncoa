function [ fitness ] = rnn_ga_fitness_function( weight_vector, train_inputwindow, train_ret, threshholds )

% ====== Assertions for MATLAB Coder Compatibility ====== %

assert(isa(weight_vector,'double'));
assert(isreal(weight_vector));

assert(isa(train_inputwindow,'double'));
assert(isreal(train_inputwindow));

assert(isa(train_ret,'double'));
assert(isreal(train_ret));

assert(isa(threshholds,'double'));
assert(isreal(threshholds));

assert ( all(size(threshholds)==[1,2]));

assert ( all(size(weight_vector)>=[1,1]));
assert ( all(size(weight_vector)<=[1,Inf]));

assert ( all(size(train_inputwindow)>=[1,1]));
assert ( all(size(train_inputwindow)<=[Inf,Inf]));

assert ( all(size(train_ret)>=[1,1]));
assert ( all(size(train_ret)<=[Inf,1]));

% ====== Main Code ====== %

m_numel = numel(weight_vector)/2;

side_length = sqrt(m_numel);

w_plus = reshape(weight_vector(1:m_numel),side_length,side_length);
w_minus = reshape(weight_vector(m_numel+1:end),side_length,side_length);

training_predictions = test_rnn(2, train_inputwindow, w_plus, w_minus);
bipolar_training_predictions = training_predictions(:,1) - training_predictions(:,2);

profit = profit_calc(train_ret(end+1-numel(bipolar_training_predictions):end),decision_maker(bipolar_training_predictions,threshholds),1);

fitness = profit(end);

end

