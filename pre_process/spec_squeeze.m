function [spec, success] = spec_squeeze(signal, ratio, win_len, target_cols)
%% Interface of spectrum 'squeeze' procedure
[signal, success] = time_squeeze(signal, ratio, win_len);
spec = spectro(signal);
if size(spec, 2) ~= target_cols
    spec = imresize(spec, [size(spec, 1), target_cols]);
end
end

function [signal, success] = time_squeeze(signal, ratio, win_len)
% Dropout 6% time domain energy, trying to reach dropout rate of ratio
len = size(signal, 1);
len2 = floor(len * ratio);
[~, e] = stzcr(signal, win_len);
cum = cumsum(e);
m = max(cum);
start_ = find(cum > m * 0.03, 1);
end_ = find(cum < m * 0.97, 1, 'last');
  
if end_ - start_ > len2
    warning('Unable to determine ends with energy by %f%%', (end_-start_)/len2*100-100);
    start_ = find(cum > m * 0.01, 1);
    end_ = find(cum < m * 0.99, 1, 'last');
    signal = signal(start_ : end_);
    success = 0;
    return;
end

mid = (start_ + end_) / 2;
start_ = floor(mid - len2/2);
end_ = floor(mid + len2/2);
if start_ < 1
    start_ = 1;
    end_ = len2-1;
end
if end_ > len
    start_ = len - len2 + 1;
    end_ = len;
end
signal = signal(start_:end_);
success = 1;
end


function signal = drop_low(signal, eng, len)
[~,i] = sort(eng);
signal = signal(sort(i(end-len+1:end)));
end
