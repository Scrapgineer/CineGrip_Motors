#include <Arduino.h>
#include <Wire.h>
#include <SimpleFOC.h> 

const int MPU_ADDR = 0x68;

// Sensor variables
float angle_y = 0; // Pitch only
int16_t raw_ax, raw_ay, raw_az, raw_gy;
unsigned long pre_interval;
float dt;

// MOTOR Config
BLDCMotor motor = BLDCMotor(7); 
BLDCDriver3PWM driver = BLDCDriver3PWM(25, 26, 27, 32);

// Smooth filter
LowPassFilter filter_angle = LowPassFilter(0.05);

void setup () {
  Serial.begin(115200);
  
  Wire.begin(21, 22); 
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0);
  Wire.endTransmission(true);

  driver.voltage_power_supply = 11.1; 
  driver.init();
  motor.linkDriver(&driver);
  
  motor.voltage_limit = 3.0;   
  
  motor.controller = MotionControlType::angle_openloop;

  motor.init();

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
    Wire.read()<<8 | Wire.read();
    raw_gy = Wire.read()<<8 | Wire.read();
  }

  unsigned long now = millis();
  dt = (now - pre_interval) / 1000.0;
  pre_interval = now;

  float acc_x = raw_ax / 16384.0;
  float acc_y = raw_ay / 16384.0;
  float acc_z = raw_az / 16384.0;
  float acc_angle_y = atan2(-acc_x, sqrt(acc_y*acc_y + acc_z*acc_z)) * 180 / PI;
  float gyro_rate_y = raw_gy / 131.0;

  angle_y = 0.98 * (angle_y + gyro_rate_y * dt) + 0.02 * acc_angle_y;
  
  float target_position = -angle_y * (PI / 180.0);
  float smooth_position = filter_angle(target_position);

  motor.move(smooth_position);

  static unsigned long timer_print = 0;
  if (millis() - timer_print > 100){
    Serial.print("Angulo Y: "); 
    Serial.println(angle_y, 2);
    timer_print = millis();
  }
}