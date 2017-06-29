%%
config;

%%
czy_prefix = '/Users/bh/projects/signal/speech_recog/czy_rec/';
czy_saveto = '/Users/bh/projects/signal/speech_recog/czy_spec/';

SHAPE = [32, 78];

for word = words
    for i = 1:2
        try
            wav = audioread(sprintf('%s%s-%02d.dat', czy_prefix, char(word), i));
        catch
            continue;
        end
%         spec = spectro(wav);          
        spec = spec_squeeze(wav(:,1), 0.4, WIN_LEN, SHAPE(2)); 

        try
            assert(all(size(spec) == SHAPE));   % TODO
        catch
            disp(size(spec));
        end
        
        filename = strcat(czy_saveto, char(word), '.txt');
        save(filename, '-ascii', '-append', 'spec');
    end
    display(char(word))
end
save(strcat(czy_saveto, 'config.txt'), '-ascii', 'SHAPE');
