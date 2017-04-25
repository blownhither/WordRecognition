function [ zcr, eng ] = stzcr( signal, half_win )
%Short-time zero-crossing rate
len = size(signal, 1);
temp = zeros(half_win, 1);
signal = [temp; signal; temp];
zcr = zeros(len, 1);
eng = zeros(len, 1);
win_len = 2 * half_win;     % actually 2*half+1 when used in [p, p+win_len]
ham = hamming(win_len + 1);
for i = 1:len
    win = signal(i : i + win_len) .* ham;
    eng(i) = sum(win.*win);
    zcr(i) = count_zc(win);
end

end


function count = count_zc(win)
win = sign(win);
count = sum(abs([0; win] - [win; 0])) * 0.5;
count = floor(count);
end

