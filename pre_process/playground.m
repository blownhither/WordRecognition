
%% stzcr
[z,e] = stzcr(wav(:,1), WIN_LEN);
subplot(3,1,1);
plot(z);title('zero-cross');
subplot(3,1,2);
plot(e);title('energy');
subplot(3,1,3);
plot(wav(:,1));

%% halve

subplot(211);
plot(wav(:,1));
subplot(212);
plot(squeeze(wav(:,1), 0.4, WIN_LEN));


%% filter
count = 0;
for word = words
    for i = 1:20
        wav = audioread(sprintf('%s%s-%02d.dat', prefix, char(word), i));
        [~,success] = squeeze(wav(:,1), 0.6, WIN_LEN);
        count = count + success;
    end
    disp(char(word));
end
display(count / 600);

%% check
f = fopen('list.txt', 'r', 'n', 'utf-8');
line = fgetl(f);
wav = audioread(line);
SHAPE = [32, 78];
spec_squeeze(wav(:,1), 0.4, WIN_LEN, SHAPE(2));