#include <Servo.h>
Servo motor1;
int ledPin = 7; 

void setup() {
  Serial.begin(9600);  //python tarafında hız aynı olacak 
  pinMode(ledPin, OUTPUT);
  motor1.attach(9); 
}

void loop() {
  if (Serial.available()) {
    int count = Serial.parseInt(); 

    if (count >= 0 && count <= 14) {
      digitalWrite(ledPin, LOW);
      motor1.write(0);
      }
    else if (count >= 15 && count <= 30) {
      digitalWrite(ledPin, HIGH); 
      motor1.write(0);           
      }
       else if (count >= 31 && count <= 45) {
      digitalWrite(ledPin, LOW);  
      motor1.write(90);          
      }
     else if (count >= 46) {
      digitalWrite(ledPin, HIGH); 
      motor1.write(90);          
    }
  }
}
