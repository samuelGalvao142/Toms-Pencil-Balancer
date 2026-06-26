#include <Arduino.h>
#include <Servo.h>
#include <EEPROM.h>
#include <Encoder.h>
 
Servo servo1;
Servo servo2;
 
String buf="";
 
const int SERVO1_PIN=5;
const int SERVO2_PIN=7;
 
Encoder encoderX(2, 4);   // servo1: interrupt pin 2 + polled pin 4
Encoder encoderY(3, 6);   // servo2: interrupt pin 3 + polled pin 6
 
const bool enc1_invert = true;
const bool enc2_invert = true;
 
const float gear_ratio = 15.0/12.0;
const float counts_per_rev = 1200.0 * gear_ratio; // 600 CPR * 2 since only one interrupt pin per encoder
 
/* PWM limits */
 
const int US_MIN=500;
const int US_MAX=2500;
 
/* calibration angles */
 
const int CAL_POINTS = 19;
float cal_angles_1[CAL_POINTS] = {49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, 104, 109, 114, 119, 124, 129, 134, 139};
float cal_angles_2[CAL_POINTS] = {52, 57, 62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117, 122, 127, 132, 137, 142};
 
/* lookup tables (microseconds) */
 
float servo1_map[CAL_POINTS];
float servo2_map[CAL_POINTS];
 
/* auto calibration settings */
const float auto_cal_tolerance = 0.5; // degrees
const float auto_cal_step = 5.0;
const int auto_cal_settle = 50; // milliseconds
const int auto_cal_iter = 100;
 
int enc1_origin = 0;
int enc2_origin = 0;
 
bool origin_set = false;
 
enum Mode{
    MODE_IDLE,
    MODE_CAL,
    MODE_EXP
};
 
Mode mode=MODE_IDLE;
 
int cal_index=0;
 
/* clamp */
 
float clamp_us(float u){
 
    if(u<US_MIN) return US_MIN;
    if(u>US_MAX) return US_MAX;
    return u;
}
 
bool invalid_map(float *map) {
 
    for(int i=0;i<CAL_POINTS;i++){
        if(isnan(map[i]) || map[i] < US_MIN || map[i] > US_MAX)
            return true;
    }
 
    return false;
}
 
/* interpolation */
 
float interp(float theta, float *angles, float *map){
 
    if(theta <= angles[0]){
        float a = angles[0];
        float b = angles[1];
        float alpha = (theta - a) / (b - a);
        return map[0] + alpha * (map[1] - map[0]);
    }
 
    if(theta >= angles[CAL_POINTS-1]){
        float a = angles[CAL_POINTS-2];
        float b = angles[CAL_POINTS-1];
        float alpha = (theta - a) / (b - a);
        return map[CAL_POINTS-2] + alpha * (map[CAL_POINTS-1] - map[CAL_POINTS-2]);
    }
 
    for(int i=0;i<CAL_POINTS-1;i++){
        float a=angles[i];
        float b=angles[i+1];
 
        if(theta>=a && theta<=b){
            float alpha=(theta-a)/(b-a);
            return map[i] + alpha*(map[i+1]-map[i]);
        }
    }
 
    return map[0];
}
 
/* move servos */
 
void move_servos(float d1,float d2){
 
    float us1=clamp_us(interp(d1,cal_angles_1, servo1_map));
    float us2=clamp_us(interp(d2,cal_angles_2, servo2_map));
 
    Serial.print("MOVED TO: ");
    Serial.print("S1=");
    Serial.print(us1);
    Serial.print(" S2=");
    Serial.println(us2);
 
    servo1.writeMicroseconds(us1);
    servo2.writeMicroseconds(us2);
}
 
/* calibration jogging */
 
void jog_servo(int id,float delta){
 
    if(id==1){
 
        servo1_map[cal_index]+=delta;
        servo1_map[cal_index]=clamp_us(servo1_map[cal_index]);
 
        servo1.writeMicroseconds(servo1_map[cal_index]);
 
        Serial.print("S1=");
        Serial.println(servo1_map[cal_index]);
    }
 
    if(id==2){
 
        servo2_map[cal_index]+=delta;
        servo2_map[cal_index]=clamp_us(servo2_map[cal_index]);
 
        servo2.writeMicroseconds(servo2_map[cal_index]);
 
        Serial.print("S2=");
        Serial.println(servo2_map[cal_index]);
    }
}
 
/* move to calibration target */
 
void goto_cal_point(){
 
    float target1 = cal_angles_1[cal_index];
    float target2 = cal_angles_2[cal_index];
 
    float us1 = interp(target1, cal_angles_1, servo1_map);
    float us2 = interp(target2, cal_angles_2, servo2_map);
 
    servo1.writeMicroseconds(us1);
    servo2.writeMicroseconds(us2);
 
    Serial.print("CAL ");
    Serial.print(target1);
    Serial.print("and ");
    Serial.println(target2);
}
 
void set_origin(){
    enc1_origin = encoderX.read();
    enc2_origin = encoderY.read();
 
    origin_set = true;
 
    Serial.println("ORIGIN SET");
    Serial.print("Encoder X = ");
    Serial.println(enc1_origin);
    Serial.print("Encoder Y = ");
    Serial.println(enc2_origin);
}
 
const float S1_origin_deg = 94.0;
const float S2_origin_deg = 97.0;
 
float enc1_deg(){
    float raw = ((float)(encoderX.read() - enc1_origin)
            / counts_per_rev) * 360.0;
    raw = enc1_invert ? -raw : raw;
    return raw + S1_origin_deg;  // shifts so center = 94 degrees
}
 
float enc2_deg(){
    float raw = ((float)(encoderY.read() - enc2_origin)
            / counts_per_rev) * 360.0;
    raw = enc2_invert ? -raw : raw;
    return raw + S2_origin_deg;  // shifts so center = 97 degrees
}
 
void auto_calibration(){
 
    if(!origin_set){
        Serial.println("ERROR: Set origin first");
        return;
    }
 
    Serial.println("AUTO CAL START");
 
    for(int i=0;i<CAL_POINTS;i++){
 
        Serial.print("Point ");
        Serial.println(i);
 
        float target1 = cal_angles_1[i];
        float target2 = cal_angles_2[i];
 
        float pwm1 = (i == 0) ? servo1_map[0] : servo1_map[i-1];
        float pwm2 = (i == 0) ? servo2_map[0] : servo2_map[i-1];
 
        // ---- SERVO 1 ----
        for(int k=0;k<auto_cal_iter;k++){
 
            servo1.writeMicroseconds(pwm1);
            delay(auto_cal_settle);
 
            float error = target1 - enc1_deg();
 
            Serial.print("S1 error=");
            Serial.print(error);
            Serial.print(" pwm=");
            Serial.println(pwm1);
 
            if(abs(error) < auto_cal_tolerance) break;
 
            float step = constrain(error * 3.0, -15.0, 15.0);
            pwm1 += step;
            pwm1 = constrain(pwm1, 1000, 1600);  // S1 physical limits
        }
 
        servo1_map[i] = pwm1;
 
        // ---- SERVO 2 ----
        for(int k=0;k<auto_cal_iter;k++){
 
            servo2.writeMicroseconds(pwm2);
            delay(auto_cal_settle);
 
            float error = target2 - enc2_deg();
 
            Serial.print("S2 error=");
            Serial.print(error);
            Serial.print(" pwm=");
            Serial.println(pwm2);
 
            if(abs(error) < auto_cal_tolerance) break;
 
            float step = constrain(error * 3.0, -15.0, 15.0);
            pwm2 += step;
            pwm2 = constrain(pwm2, 1200, 1800);  // S2 physical limits
        }
 
        servo2_map[i] = pwm2;
 
        Serial.print("S1 PWM = ");
        Serial.println(pwm1);
        Serial.print("S2 PWM = ");
        Serial.println(pwm2);
    }
 
    EEPROM.put(0, servo1_map);
    EEPROM.put(sizeof(servo1_map), servo2_map);
 
    Serial.println("AUTO CAL COMPLETE");
}
/* command parser */
 
void handle_command(String cmd){
 
    cmd.trim();
 
    if(cmd=="T" && mode==MODE_CAL){
 
        if(cal_index<CAL_POINTS-1) cal_index++;
 
        goto_cal_point();
        return;
    }
 
    if(cmd=="R" && mode==MODE_CAL){
 
        if(cal_index>0) cal_index--;
 
        goto_cal_point();
        return;
    }
 
    int c1=cmd.indexOf(',');
    String head=c1>0?cmd.substring(0,c1):cmd;
 
    if(head=="MODE"){
 
        String arg=cmd.substring(c1+1);
 
        if(arg=="CAL"){
 
            mode=MODE_CAL;
            cal_index=0;
 
            goto_cal_point();
        }
 
        if(arg=="IDLE"){
 
            mode=MODE_IDLE;
        }
 
        if(arg=="EXP"){
 
            mode=MODE_EXP;
        }
 
        return;
    }
 
    if(head=="CMD" && mode==MODE_EXP){
 
        int c2=cmd.indexOf(',',c1+1);
 
        float d1=cmd.substring(c1+1,c2).toFloat();
        float d2=cmd.substring(c2+1).toFloat();
 
        move_servos(d1,d2);
 
        return;
    }
 
    if(head=="JOG" && mode==MODE_CAL){
 
        int c2=cmd.indexOf(',',c1+1);
 
        int id=cmd.substring(c1+1,c2).toInt();
        float delta=cmd.substring(c2+1).toFloat();
 
        jog_servo(id,delta);
 
        return;
    }
 
    if(cmd=="SAVE"){
 
        EEPROM.put(0,servo1_map);
        EEPROM.put(sizeof(servo1_map),servo2_map);
 
        Serial.println("CAL SAVED");
 
        return;
    }
 
    if(cmd == "ENC_ZERO"){
        encoderX.write(0);
        encoderY.write(0);
        Serial.println("ENC ZEROED");
        return;
    }
 
    if(cmd=="SET_ORIGIN"){
        set_origin();
        return;
    }
 
    if (cmd == "AUTO_CAL"){
        auto_calibration();
        return;
    }
 
    if(cmd=="RESET_CAL"){
        float defaults1[CAL_POINTS] = {1000, 1033, 1066, 1100, 1133, 1166, 1200, 1233, 1266, 1300, 1333, 1366, 1400, 1433, 1466, 1500, 1533, 1566, 1600};
        float defaults2[CAL_POINTS] = {1200, 1233, 1266, 1300, 1333, 1366, 1400, 1433, 1466, 1500, 1533, 1566, 1600, 1633, 1666, 1700, 1733, 1766, 1800};
    for(int i=0;i<CAL_POINTS;i++){
        servo1_map[i] = defaults1[i];
        servo2_map[i] = defaults2[i];
    }
    EEPROM.put(0, servo1_map);
    EEPROM.put(sizeof(servo1_map), servo2_map);
    Serial.println("CAL RESET");
    return;
  }
}
 
/* setup */
 
void setup(){
 
    Serial.begin(115200);
 
    servo1.attach(SERVO1_PIN);
    servo2.attach(SERVO2_PIN);
 
    encoderX.write(0);
    encoderY.write(0);
 
    EEPROM.get(0,servo1_map);
    EEPROM.get(sizeof(servo1_map),servo2_map);
 
    if(invalid_map(servo1_map) || invalid_map(servo2_map)){
 
    Serial.println("EEPROM invalid → initializing defaults");
    float defaults1[CAL_POINTS] = {1000, 1033, 1066, 1100, 1133, 1166, 1200, 1233, 1266, 1300, 1333, 1366, 1400, 1433, 1466, 1500, 1533, 1566, 1600};
    float defaults2[CAL_POINTS] = {1200, 1233, 1266, 1300, 1333, 1366, 1400, 1433, 1466, 1500, 1533, 1566, 1600, 1633, 1666, 1700, 1733, 1766, 1800};
 
    for(int i=0;i<CAL_POINTS;i++){
        servo1_map[i] = defaults1[i];
        servo2_map[i] = defaults2[i];
        }
    }
 
    Serial.println("READY");
}
 
/* main loop */
 
void loop(){
 
    unsigned long now = millis();
    static unsigned long lastPrint = 0;
 
    if(now - lastPrint > 100){
        float angle_x = enc1_deg();
        float angle_y = enc2_deg();
        Serial.print("ENC,");
        Serial.print(angle_x, 6);
        Serial.print(",");
        Serial.println(angle_y, 6);
        lastPrint = now;
    }
 
    while(Serial.available()){
        char c=Serial.read();
        if(c=='\r') continue;
        if(c=='\n'){
            handle_command(buf);
            buf="";
        }
        else buf+=c;
    }
}