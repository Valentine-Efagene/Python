import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import Slider

frequency = 20
f_max = 100
fp_max = 50
fs = 2 * frequency # Sampling frequency

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.margins(x=0)
ax.set_ylabel('Perceived Frequency = ' + str(np.round(np.abs(frequency-fs*np.round(frequency/fs)), 2)) + 'Hz')
ax.set_xlabel('Input Frequency = ' + str(np.round(frequency, 2)) + 'Hz')
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

f = np.arange(0, f_max, 0.1)
fp = np.abs(f-fs*np.round(f/fs))
signal, = plt.plot(f, fp, lw=2)
plt.title('FOLDING DIAGRAM')
plt.xlim(0, f_max)
plt.ylim(0, fp_max)
point = ax.scatter(frequency, np.abs(frequency-fs*np.round(frequency/fs)), marker='o', color='black')

axcolor = 'lightgoldenrodyellow'
axSR = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

s_freq = Slider(axfreq, 'Freq', 0.1, f_max, valinit=frequency, valstep=0.1)
s_sr = Slider(axSR, 'Sampling Freq', 0.1, 100, valinit=2 * frequency, valstep=0.1)

def update(val):
    fs = s_sr.val
    frequency = s_freq.val

    plt.title('Sampling Frequency = ' + str(np.round(fs, 2)))
    point.set_offsets([frequency, np.abs(frequency-fs*np.round(frequency/fs))])
    signal.set_ydata(np.abs(f-fs*np.round(f/fs)))
    ax.set_ylabel('Perceived Frequency = ' + str(np.round(np.abs(frequency-fs*np.round(frequency/fs)), 2)) + 'Hz')
    ax.set_xlabel('Input Frequency = ' + str(np.round(frequency, 2)) + 'Hz')
    fig.canvas.draw_idle()

s_sr.on_changed(update)
s_freq.on_changed(update)

plt.show()