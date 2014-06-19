function [ inputwindow, outputtarget ] = createtrainingdata( ts , windowsize )

outputtarget = [max(0,ts(windowsize+1:end))',abs(min(0,ts(windowsize+1:end)))'];

ts_len = size(ts,2);

inputwindow = zeros(ts_len-windowsize,windowsize*size(ts,1));

for i=(windowsize+1):ts_len
    inputwindow(i-windowsize,:) = reshape(ts(:,i-windowsize:i-1),1,[])';
end

%inputwindow = [inputwindow,-1*inputwindow];
inputwindow = [max(0,inputwindow),abs(min(0,inputwindow))];

