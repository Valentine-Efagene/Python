import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

sample_time = 1
allowance = 4
amplitude = 5
frequency = 8
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

t = np.arange(0.0, sample_time, 0.001)
ts = np.linspace(0.0, sample_time, 2 * frequency + allowance)

s = amplitude * np.sin(2 * np.pi * frequency * t)
q = amplitude * np.sin(2 * np.pi * frequency * ts)

signal, = plt.plot(t, s, lw=2)
sampling, = plt.plot(ts, q, lw=2, marker='o')
ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
axSR = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axfreq = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=frequency, valstep=0.1)
sr = Slider(axSR, 'Sample Rate', 0.1, 70, valinit=2 * frequency + allowance, valstep=0.1)
ax.set_xlabel('Time; Frequency = ' + str(frequency) + 'Hz')
ax.set_ylabel('Amplitude')

def update(val):
    frequency = sfreq.val
    signal.set_ydata(amplitude*np.sin(2*np.pi*frequency*t))

    r = sr.val + 1 # Point zero takes one
    ts = np.linspace(0.0, sample_time, r)
    sampling.set_xdata(ts)
    sampling.set_ydata(amplitude*np.sin(2*np.pi*frequency*ts))
    ax.set_xlabel('Time; Frequency = ' + str(frequency) + 'Hz')
    fig.canvas.draw_idle()


sr.on_changed(update)
sfreq.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sr.reset()
    sfreq.reset()

button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'darkorange', 'green'), active=0)


def colorfunc(label):
    sampling.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()
