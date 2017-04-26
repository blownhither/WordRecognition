%% find data
!sh find_data.sh

%% setting up
config;

%% read data
f = fopen('list.txt', 'r', 'n', 'utf-8');
line = fgetl(f);
count = 0;
miss = 0;
SHAPE = [32, 78];
while ischar(line)
    try
        wav = audioread(line);
        
        spec = spec_squeeze(wav(:,1), 0.4, WIN_LEN, SHAPE(2));
        try
            assert(all(size(spec) == SHAPE));   % TODO
        catch
            disp(size(spec));
        end
        filename = strcat(save_to, char(word), '.txt');
        save(filename, '-ascii', '-append', 'spec');
        
        count = count + 1;
    catch
        miss = miss + 1;
    end
    line = fgetl(f);
    if mod(count, 100) == 0
        disp(count);
        p = ftell(f);
        fclose('all');      % close files that audioread forgets to close!
        f = fopen('list.txt', 'r', 'n', 'utf-8');
        fseek(f, p, 0);
    end
end
fclose('all');
display(sprintf('Read %d files.', count));
display(sprintf('Ignored %d files.', miss));

