# Simple demo of the LSM9DS1 accelerometer, magnetometer, gyroscope.
# Will print the acceleration, magnetometer, and gyroscope values every second.
import time
import board
import busio
import adafruit_lsm9ds1
import numpy as np
import math

# # I2C connection:
# i2c = busio.I2C(board.SCL, board.SDA)
# sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)


class IMU:
    def __init__(self):
        # I2C connection:
        # SPI connection:
        # from digitalio import DigitalInOut, Direction
        # spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
        # csag = DigitalInOut(board.D5)
        # csag.direction = Direction.OUTPUT
        # csag.value = True
        # csm = DigitalInOut(board.D6)
        # csm.direction = Direction.OUTPUT
        # csm.value = True
        # sensor = adafruit_lsm9ds1.LSM9DS1_SPI(spi, csag, csm)
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_lsm9ds1.LSM9DS1_I2C(self.i2c)
        # Calibration Parameters
        self.x_gyro_calibration = 0
        self.y_gyro_calibration = 0
        self.z_gyro_calibration = 0
        self.roll_calibration = 0
        self.pitch_calibration = 0
        self.yaw_calibration = 0
        # IMU Parameters: acc (x,y,z), gyro(x,y,z)
        self.imu_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Time in seconds
        self.prev_time = time.time()
        # IMU timer
        self.imu_diff = 0
        # Gyroscope integrals for filtering
        self.roll_int = 0
        self.pitch_int = 0
        self.yaw_int = 0
        # Complementary Filter Coefficient
        self.comp_filter = 0.02
        # Filtered RPY
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # CALIBRATE
        self.calibrate_imu()

        print("IMU Calibrated!")

    def calibrate_imu(self):
        """
        """
        # Reset calibration params
        self.x_gyro_calibration = 0
        self.y_gyro_calibration = 0
        self.z_gyro_calibration = 0
        self.roll_calibration = 0
        self.pitch_calibration = 0
        self.yaw_calibration = 0

        sum_xg = 0
        sum_yg = 0
        sum_zg = 0
        sum_xa = 0
        sum_ya = 0
        sum_za = 0
        sum_roll = 0
        sum_pitch = 0
        sum_yaw = 0

        num_calibrations = 1000

        for i in range(num_calibrations):

            self.read_imu()

            sum_xg += self.imu_data[0]
            sum_yg += self.imu_data[1]
            sum_zg += self.imu_data[2]

            sum_xa += self.imu_data[3]
            sum_ya += self.imu_data[4]
            sum_za += self.imu_data[5]

            # Y,Z accelerations make up roll
            sum_roll += (math.atan2(self.imu_data[3],
                                    self.imu_data[5])) * 180.0 / np.pi
            # X,Z accelerations make up pitch
            sum_pitch += (math.atan2(self.imu_data[4],
                                     self.imu_data[5])) * 180.0 / np.pi
            # Y, X accelerations make up yaw
            sum_yaw += (math.atan2(self.imu_data[3],
                                   self.imu_data[4])) * 180.0 / np.pi

        # Average values for calibration
        self.x_gyro_calibration = sum_xg / float(num_calibrations)
        self.y_gyro_calibration = sum_yg / float(num_calibrations)
        self.z_gyro_calibration = sum_zg / float(num_calibrations)
        self.roll_calibration = sum_roll / float(num_calibrations)
        self.pitch_calibration = sum_pitch / float(num_calibrations)
        self.yaw_calibration = sum_yaw / float(num_calibrations)

    def read_imu(self):
        """
        """
        accel_x, accel_y, accel_z = self.sensor.acceleration
        # mag_x, mag_y, mag_z = self.sensor.magnetic
        gyro_x, gyro_y, gyro_z = self.sensor.gyro
        # temp = self.sensor.temperature

        # Populate imu data list
        # Gyroscope Values (Degrees/sec)
        self.imu_data[0] = gyro_x - self.x_gyro_calibration
        self.imu_data[1] = gyro_y - self.y_gyro_calibration
        self.imu_data[2] = gyro_z - self.z_gyro_calibration
        # Accelerometer Values (m/s^2)
        self.imu_data[3] = accel_x
        self.imu_data[4] = accel_y
        self.imu_data[5] = accel_z

    def filter_rpy(self):
        """
        """
        # Get Current Time in seconds
        current_time = time.time()
        self.imu_diff = current_time - self.prev_time
        # Set new previous time
        self.prev_time = current_time
        # Catch rollovers
        if self.imu_diff < 0:
            self.imu_diff = 0

        # Complementary filter for RPY
        # TODO: DOUBLE CHECK THIS!!!!!!!
        roll_gyro_delta = self.imu_data[1] * self.imu_diff
        pitch_gyro_delta = self.imu_data[0] * self.imu_diff
        yaw_gyro_delta = self.imu_data[2] * self.imu_diff

        self.roll_int += roll_gyro_delta
        self.pitch_int += pitch_gyro_delta
        self.yaw_int += yaw_gyro_delta

        # RPY from Accelerometer
        # Y,Z accelerations make up roll
        roll_a = (math.atan2(self.imu_data[3], self.imu_data[5])
                  ) * 180.0 / np.pi - self.roll_calibration
        # X,Z accelerations make up pitch
        pitch_a = (math.atan2(self.imu_data[4], self.imu_data[5])
                   ) * 180.0 / np.pi - self.pitch_calibration
        # Y, X accelerations make up yaw
        yaw_a = (math.atan2(self.imu_data[3], self.imu_data[4])
                 ) * 180.0 / np.pi - self.yaw_calibration

        # Calculate Filtered RPY
        self.roll = roll_a * self.comp_filter + (1 - self.comp_filter) * (
            roll_gyro_delta + self.roll)
        self.pitch = pitch_a * self.comp_filter + (1 - self.comp_filter) * (
            pitch_gyro_delta + self.pitch)
        self.yaw = yaw_a * self.comp_filter + (1 - self.comp_filter) * (
            pitch_gyro_delta + self.yaw)
        # self.roll = roll_a
        # self.pitch = pitch_a
        # self.yaw = yaw_a


if __name__ == "__main__":
    imu = IMU()

    while True:

        imu.filter_rpy()
        imu.read_imu()

        print("ROLL: {} \t PICH: {} \t YAW: {} \n".format(
            imu.roll, imu.pitch, imu.yaw))