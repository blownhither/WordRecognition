%% load configuration
config;

%% mainloop
ar = audiorecorder(FREQ, 16, 2);
SHAPE = [32, 78];
while(true) 
    disp('Press ENTER to start recording: (2s)');
    input('');
    recordblocking(ar, 2);
    disp('Done');
    data = getaudiodata(ar);
    
    data = data(100:end, :);
    
    tic;
    spec = spec_squeeze(data(:,1), 0.4, WIN_LEN, SHAPE(2));
    toc;
    play(ar);

    save('../working/working.txt', '-ascii', 'spec');
end