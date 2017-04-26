%% find data
!sh ../find_data.sh

%% read data
f = fopen('../list.txt', 'r', 'n', 'utf-8');
line = fgetl(f);
count = 0;
miss = 0;
while ischar(line)
    try
        wav = audioread(strcat('../', line));
        count = count + 1;
    catch
        miss = miss + 1;
    end
    line = fgetl(f);
    if mod(count, 100) == 0
        disp(count);
        p = ftell(f);
        fclose('all');      % close files that audioread forgets to close!
        f = fopen('../list.txt', 'r', 'n', 'utf-8');
        fseek(f, p, 0);
    end
end
fclose(f);
display(sprintf('Read %d files.', count));
display(sprintf('Ignored %d files.', miss));
