%% Setting up
prefix = '/Users/bh/projects/signal/speech_recog/rec/14307130033-';
words = {'??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', 'Speech', 'Voice', 'Sound', 'Happy', 'Lucky', 'Data', 'Recognition', 'File', 'Open', 'Close', 'Start', 'Stop', 'Network', 'Computer', 'China'};

save_to = '/Users/bh/projects/signal/speech_recog/spec/';

FREQ = 8192;
WIN_LEN = floor(FREQ * 0.02);
N_OVERLAP = floor(WIN_LEN * 0.5);