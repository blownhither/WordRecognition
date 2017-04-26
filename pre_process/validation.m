%% find data
!sh find_data.sh

%% setting up
config;

%% read data
fclose('all');
f = fopen('list.txt', 'r', 'n', 'utf-8');
line = '';
count = 0;
miss = 0;
SHAPE = [32, 78];

% TODO: 
% skip = 0;
% for i = 1:skip
%     fgetl(f);
% end
% count = skip;

while ischar(line)
    line = fgetl(f);
    
    if mod(count + miss, 50) == 0
        disp(count);
        p = ftell(f);
        fclose('all');      % close files that audioread forgets to close!
        f = fopen('list.txt', 'r', 'n', 'utf-8');
        fseek(f, p, 0);
    end
    
    try
        wav = audioread(line);
    catch
        warning('Error reading %s', line);
        miss = miss + 1;
        continue;
    end
    spec = spec_squeeze(wav(:,1), 0.4, WIN_LEN, SHAPE(2));
    try
        assert(all(size(spec) == SHAPE));   % TODO
    catch
        warning(mat2str(size(spec)));
    end
    position = find(line == '_' | line == '-');
    filename = strcat(save_to, line(position(end-1)+1:position(end)-1), '.txt');
    save(filename, '-ascii', '-append', 'spec');
    count = count + 1;

end
fclose('all');
display(sprintf('Read %d files.', count));
display(sprintf('Ignored %d files.', miss));


