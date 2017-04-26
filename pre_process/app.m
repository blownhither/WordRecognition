%% Setting up
prefix = '/Users/bh/projects/signal/speech_recog/rec/14307130033-';
words = {'����', '����', '����', '�ź�', '����', 'ʶ��', '����', '�й�', '����', '����', '�Ϻ�', '����', '����', '����', '����', 'Speech', 'Voice', 'Sound', 'Happy', 'Lucky', 'Data', 'Recognition', 'File', 'Open', 'Close', 'Start', 'Stop', 'Network', 'Computer', 'China'};

save_to = '/Users/bh/projects/signal/speech_recog/spec/';

FREQ = 8192;
WIN_LEN = floor(FREQ * 0.02);
N_OVERLAP = floor(WIN_LEN * 0.5);


%% process
SHAPE = [32, 78];

for word = words
    for i = 1:20
        wav = audioread(sprintf('%s%s-%02d.dat', prefix, char(word), i));
%         spec = spectro(wav);          
        spec = spec_squeeze(wav(:,1), 0.4, WIN_LEN, SHAPE(2)); 

        try
            assert(all(size(spec) == SHAPE));   % TODO
        catch
            disp(size(spec));
        end
        
        filename = strcat(save_to, char(word), '.txt');
        save(filename, '-ascii', '-append', 'spec');
    end
    display(char(word))
end
save(strcat(save_to, 'config.txt'), '-ascii', 'SHAPE');

% disp(spec);

%% demo
word = 'China';
wav = audioread(sprintf('%s%s-%02d.dat', prefix, char(word), 1));
freq = 8192;
win_len = floor(8192 * 0.02);
n_overlap = floor(win_len * 0.5);
n_freq = 6400;
spectrogram(wav(:,1), win_len, n_overlap, n_freq, freq);
% imshow(spec, []);
