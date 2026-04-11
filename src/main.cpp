#include <Arduino.h>
#include <Wire.h>
#include <SimpleFOC.h> 

const int MPU_ADDR = 0x68;

// Sensor variables
float angle_y = 0; // Tilt
float angle_x = 0; // Pan
int16_t raw_ax, raw_ay, raw_az, raw_gx,raw_gy, raw_gz;
unsigned long pre_interval;
float dt;

// MOTOR Config
BLDCMotor motor_y = BLDCMotor(7); 
BLDCDriver3PWM driver_y = BLDCDriver3PWM(32, 33, 25, 26);
LowPassFilter filter_angle_y = LowPassFilter(0.05);

BLDCMotor motor_x = BLDCMotor(7); 
BLDCDriver3PWM driver_x = BLDCDriver3PWM(27, 14, 16, 17);
LowPassFilter filter_angle_x = LowPassFilter(0.05);

void setup () {
  Serial.begin(115200);
  
  Wire.begin(21, 22); 
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0);
  Wire.endTransmission(true);

  //Driver Y
  driver_y.voltage_power_supply = 11.1; 
  driver_y.init();
  motor_y.linkDriver(&driver_y);
  motor_y.voltage_limit = 3.0;   
  motor_y.controller = MotionControlType::angle_openloop;
  motor_y.init();

  //Driver X
  driver_x.voltage_power_supply = 11.1; 
  driver_x.init();
  motor_x.linkDriver(&driver_x);
  motor_x.voltage_limit = 3.0;   
  motor_x.controller = MotionControlType::angle_openloop;
  motor_x.init();

  Serial.println("Calibrando... deja el sensor QUIETO 2 segundos.");
  delay(2000);
  pre_interval = millis();
}

void loop () {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, 14, 1);

  if(Wire.available() == 14) {
    raw_ax = Wire.read()<<8 | Wire.read();
    raw_ay = Wire.read()<<8 | Wire.read();
    raw_az = Wire.read()<<8 | Wire.read();
    Wire.read()<<8 | Wire.read();
    raw_gx = Wire.read()<<8 | Wire.read();
    raw_gy = Wire.read()<<8 | Wire.read();
    raw_gz = Wire.read()<<8 | Wire.read();
  }

  unsigned long now = millis();
  dt = (now - pre_interval) / 1000.0;
  pre_interval = now;

  float acc_x = raw_ax / 16384.0;
  float acc_y = raw_ay / 16384.0;
  float acc_z = raw_az / 16384.0;

  float gyro_rate_x = raw_gx / 131.0;
  float gyro_rate_y = raw_gy / 131.0;

  float acc_angle_y = atan2(-acc_x, sqrt(acc_y*acc_y + acc_z*acc_z)) * 180 / PI;
  float acc_angle_x = atan2(acc_y, sqrt(acc_x*acc_x + acc_z*acc_z)) * 180 / PI;

  angle_y = 0.98 * (angle_y + gyro_rate_y * dt) + 0.02 * acc_angle_y;
  angle_x = 0.98 * (angle_x + gyro_rate_x * dt) + 0.02 * acc_angle_x;

  float target_position_y = -angle_y * (PI / 180.0);
  float target_position_x = -angle_x * (PI / 180.0);

  float smooth_position_y = filter_angle_y(target_position_y);
  float smooth_position_x = filter_angle_x(target_position_x);

  motor_y.move(smooth_position_y);
  motor_x.move(smooth_position_x);

  static unsigned long timer_print = 0;
  if (millis() - timer_print > 100){
    Serial.print("Tilt Y: "); 
    Serial.print(angle_y, 2);
    Serial.print("Pan X: "); 
    Serial.println(angle_x, 2);
    timer_print = millis();
  }
}