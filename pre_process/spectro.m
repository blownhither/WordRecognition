function [spec] = spectro(wav, win_len, n_overlap, n_freq, max_freq)
%SPECTRO Simplified spectrogram
%   When used with no further understanding of SPECTROGRAM, simple input wav.
%   Return spectrogram[n_freq][time], in an ascending order.
%   Default settings applies to human voice recognization.

if nargin < 5
    max_freq = 8192;                        % recording freqency
end
if nargin < 4
    n_freq = 64;                            % freq resolution 
end
if nargin < 2
    win_len = floor(max_freq * 0.02);       % 20ms
    n_overlap = floor(win_len * 0.5);       % half_window
end

if size(wav, 2) > 1
    wav = wav(:,1);                         % only one channel
end

spec = spectrogram(wav, win_len, n_overlap, n_freq, max_freq);
spec = abs(spec(2:end,:));                  % dropping lowest freq to align 
end

