function [ balance ] = profit_calc( ret_series, decisions, can_short )

if can_short == 0
    decisions = max(decisions,0);
end

combined_ret_series = ret_series .* decisions;

balance = [1;cumprod(combined_ret_series+1)];

end