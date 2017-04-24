%% Setting up
prefix = '/Users/bh/projects/signal/speech_recog/rec/14307130033-';
words = {'数字', '语音', '话音', '信号', '分析', '识别', '数据', '中国', '北京', '背景', '上海', '商行', '复旦', '网络', '电脑', 'Speech', 'Voice', 'Sound', 'Happy', 'Lucky', 'Data', 'Recognition', 'File', 'Open', 'Close', 'Start', 'Stop', 'Network', 'Computer', 'China'};

save_to = '/Users/bh/projects/signal/speech_recog/spec/';

FREQ = 8192;
WIN_LEN = floor(FREQ * 0.02);
N_OVERLAP = floor(WIN_LEN * 0.5);


%% process
SHAPE = [128, 198];

for word = words
    for i = 1:20
        wav = audioread(sprintf('%s%s-%02d.dat', prefix, char(word), i));
        spec = spectro(wav);
        
        assert(all(size(spec) == SHAPE));   % TODO
        
        filename = strcat(save_to, char(word), '.txt');
        save(filename, '-ascii', '-append', 'spec');
    end
end
save(strcat(save_to, 'config.txt'), '-ascii', 'SHAPE');

disp(spec);