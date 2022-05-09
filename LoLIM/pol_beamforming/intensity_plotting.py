#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

import sys

class image_plotter:
    def __init__(self, X_range, Y_range, Z_range, image, square=False, color_range=None):
        self.X_range = X_range
        self.Y_range = Y_range
        self.Z_range = Z_range
        self.color_range = color_range

        self.square = square

        self.X_offset = int(len(self.X_range) / 4)
        self.X_width = int(len(self.X_range) / 2)
        self.Y_offset = int(len(self.Y_range) / 4)
        self.Y_width = int(len(self.Y_range) / 2)
        self.Z_offset = int(len(self.Z_range) / 4)
        self.Z_width = int(len(self.Z_range) / 2)

        self.image = image
        self.shape = image.shape
        if len(self.shape) > 3:
            print('can only plot 3D data!')
            quit()

        if color_range is None:
            self.vmin = np.min(image)
            self.vmax = np.max(image)
        else:
            self.vmin = color_range[0]
            self.vmax = color_range[1]

        self.mode = 2
        self.index = int(self.shape[self.mode] / 2)

        self.fig, self.axis = plt.subplots()

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        self.clear_and_plot()
        # self.axis.imshow( self.mode_slice(), origin='lower', vmin=self.vmin, vmax=self.vmax)

        plt.show()

        print(' '*100)

    def mode_slice(self):
        if self.mode == 0:
            return self.image[self.index, :, :].T
        if self.mode == 1:
            return self.image[:, self.index, :].T
        if self.mode == 2:
            return self.image[:, :, self.index].T

    def plot_square(self, XY_start, width_height):
        if self.square:
            self.axis.plot((XY_start[0], XY_start[0] + width_height[0]), (XY_start[1], XY_start[1]), c='k')
            self.axis.plot((XY_start[0], XY_start[0]), (XY_start[1], XY_start[1] + width_height[1]), c='k')
            self.axis.plot((XY_start[0] + width_height[0], XY_start[0] + width_height[0]),
                           (XY_start[1] + width_height[1], XY_start[1]), c='k')
            self.axis.plot((XY_start[0] + width_height[0], XY_start[0]),
                           (XY_start[1] + width_height[1], XY_start[1] + width_height[1]), c='k')

    def clear_and_plot(self):
        print('mode:', self.mode, 'index', self.index, '/', self.shape[self.mode], end='r')

        if self.mode == 0:
            title = 'X ' + str(self.X_range[self.index])
        elif self.mode == 1:
            title = 'Y ' + str(self.Y_range[self.index])
        elif self.mode == 2:
            title = 'Z ' + str(self.Z_range[self.index])

        self.axis.clear()
        self.axis.set_title(title)
        self.axis.imshow(self.mode_slice(), origin='lower', vmin=self.vmin, vmax=self.vmax)

        if self.mode == 0:
            title = 'X ' + str(self.X_range[self.index])
        elif self.mode == 1:
            title = 'Y ' + str(self.Y_range[self.index])
        elif self.mode == 2:
            self.plot_square((self.X_offset, self.Y_offset), (self.X_width, self.X_width))

        self.fig.canvas.draw()

    def on_press(self, event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'a':
            self.index -= 1
            if self.index < 0:
                self.index = self.shape[self.mode] - 1
            self.clear_and_plot()

        elif event.key == 'd':
            self.index += 1
            if self.index >= self.shape[self.mode]:
                self.index = 0
            self.clear_and_plot()

        elif event.key in ['0', '1', '2']:
            self.mode = int(event.key)
            self.index = int(self.shape[self.mode] / 2)
            self.clear_and_plot()

