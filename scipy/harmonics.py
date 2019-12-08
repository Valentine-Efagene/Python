from pylab import *

style.use('ggplot')

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
Y1 = np.sin(X)
Y2 = np.sin(2 * X)
Y3 = np.sin(3 * X)
Y4 = np.sin(4 * X)

subplot(2, 2, 1)
title('1st harmonic')
plot(X, Y1, color = 'black')

subplot(2, 2, 2)
title('2nd harmonic')
plot(X, Y2, color = 'blue')

subplot(2, 2, 3)
title('3rd harmonic')
plot(X, Y3, color = 'yellow')

subplot(2, 2, 4)
title('4th harmonic')
plot(X, Y4, color = 'red')

show()