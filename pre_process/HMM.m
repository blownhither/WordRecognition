%%
config;

%%
for word = words
    filename = strcat('Users/bh/projects/signal/speech_recog/spec/' , char(word), '.txt');
    temp = textread(filename);
end