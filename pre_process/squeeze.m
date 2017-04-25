function [signal, success] = squeeze(signal, ratio, win_len)
len = size(signal, 1);
len2 = floor(len * ratio);
[~, e] = stzcr(signal, win_len);
cum = cumsum(e);
m = max(cum);
start_ = find(cum > m * 0.03, 1);
end_ = find(cum < m * 0.97, 1, 'last');
  
if end_ - start_ > len2
%     signal2 = drop_low(signal(start_:end_), e(start_:end_), len2);
    warning('Unable to determine ends with energy by %f%%', (end_-start_)/len2*100-100);
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
