%one = agent_tester( @(x)(fetch_test_mlp(x, '01/01/08', '12/31/12', '01/01/13', '12/31/13' )), {'AAPL','GOOGL','AMZN','MSFT','BA.L','VOD.L','JNJ','XOM'}, 10,4 );
%two = agent_tester( @(x)(fetch_test_mlp(x, '01/01/08', '12/31/12', '01/01/13', '12/31/13' )), {'BLT','LLOY.L','BARC.L','GSK.L','AZN.L','DGE.L','AXP','IBM'}, 10,4 );
three = agent_tester( @(x)(fetch_test_mlp(x, '01/01/08', '12/31/12', '01/01/13', '12/31/13' )), {'EZJ.L','MT','GS','T','KO','PFE','NKE','GE'}, 10,4 );
    