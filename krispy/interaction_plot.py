from matplotlib import pyplot as plt

def draw_lines(data=None):

    class LineBuilder:
        def __init__(self):
            line, = ax.plot([], [], marker='x', linestyle='-', color='red') 
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            if event.inaxes!=self.line.axes: return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw() #_idle

        def get_xy(self):
            return self.xs, self.ys

    if data == None:
        data = [[1,2,3], [4,5,6], [7,8,9]]

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_title('click to build line segments')
    plt.imshow(data, origin='lower')
    linebuilder = LineBuilder()
    plt.show()

    while plt.fignum_exists(1): #while plot is being drawn on don't continue
        plt.pause(0.001) #Pauses to allow plot to draw, can't draw if not paused

    return linebuilder.get_xy()

