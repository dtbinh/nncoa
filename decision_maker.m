function [ decisions ] = decision_maker( bipolar_predictions, thresholds )

long_threshold = max(0,thresholds(1));
short_threshold = min(0,thresholds(2));

decisions = zeros(size(bipolar_predictions));

decisions(bipolar_predictions > long_threshold) = 1;
decisions(bipolar_predictions < short_threshold) = -1;
decisions(bipolar_predictions < long_threshold & bipolar_predictions > short_threshold) = 0;

end