import os
import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

# create the spi bus
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)

# create the cs (chip select)
cs1 = digitalio.DigitalInOut(board.D22)
cs2 = digitalio.DigitalInOut(board.D27)

# create the mcp object
mcp1 = MCP.MCP3008(spi, cs1)
mcp2 = MCP.MCP3008(spi, cs2)

# create an analog input channel on pin 0
chan_1_0 = AnalogIn(mcp1, MCP.P0)

# create an analog input channel on pin 0
chan_2_0 = AnalogIn(mcp2, MCP.P0)

print('Raw ADC Value 1: ', chan_1_0.value)
print('ADC Voltage 1: ' + str(chan_1_0.voltage) + 'V')

print('Raw ADC Value 2: ', chan_2_0.value)
print('ADC Voltage 2: ' + str(chan_2_0.voltage) + 'V')

last_read = 0  # this keeps track of the last potentiometer value
tolerance = 50  # to keep from being jittery we'll only change
# pos when the pot has moved a significant amount
# on a 16-bit ADC


def remap_range(value, in_min, in_max, out_min, out_max):
    """ remap using y = mx + b
    """
    m = float(out_max - out_min) / float(in_max - in_min)
    b = out_max - (m * in_max)

    return m * value + b

while True:
    # convert 16bit adc0 (0-65535) trim pot read into -90|90 pos level
    read_pos_1 = remap_range(chan_1_0.value, 10944, 46784, -90, 90)
    read_pos_2 = remap_range(chan_2_0.value, 10944, 46784, -90, 90)

    # set OS pos playback pos
    print('POS1 = {pos}%'.format(pos=read_pos_1))
    print('POS2 = {pos}%'.format(pos=read_pos_2))

    # hang out and do nothing for a half second
    time.sleep(0.5)