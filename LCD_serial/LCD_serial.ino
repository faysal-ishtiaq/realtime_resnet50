#include <LiquidCrystal.h>

LiquidCrystal lcd( 10, 9, 5, 4, 3, 2);
String a;

void setup() {
  
  lcd.begin(16, 2);
  Serial.begin(9600);
}

void loop() {
  while(Serial.available()) {

    a= Serial.readString();// read the incoming data as string
    
    
    lcd.clear();
    lcd.setCursor(0, 0);
    
    lcd.print(a);
    
  }
}

