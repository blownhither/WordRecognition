import pyaudio
import wave
import pylab as pl
import numpy as np
import wave
import sys
import time

def rec(name, hint):
    pl.close()

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 8192
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = name
    # WAVE_OUTPUT_FILENAME = sys.argv[1]
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* recording " + hint)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    f = wave.open(WAVE_OUTPUT_FILENAME, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0 / framerate)


    # pl.subplot(211)
    pl.plot(time, wave_data[0])
    # pl.subplot(212)
    # pl.plot(time, wave_data[1], c="g")
    pl.xlabel("time (seconds)")
    pl.show(False)


def main():
    hints = ['数字', '语音', '话音', '信号', '分析', '识别', '数据', '中国', '北京', '背景', '上海', '商行', '复旦', '网络', '电脑', 'Speech', 'Voice', 'Sound', 'Happy', 'Lucky', 'Data', 'Recognition', 'File', 'Open', 'Close', 'Start', 'Stop', 'Network', 'Computer', 'China']
    # for i, v in enumerate(hints):
    for v in hints[1:]:
        for step in range(1,20+1):
            input()
            rec('./rec/14307130033-%s-%02d.dat' % (v, step), v+str(step))


if __name__ == "__main__":
    main()