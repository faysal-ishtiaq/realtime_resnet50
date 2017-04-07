// include the library code:
#include <LiquidCrystal.h>

// initialize the library with the numbers of the interface pins
LiquidCrystal lcd( 10, 9, 5, 4, 3, 2);
String a;

void setup() {
  // set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  Serial.begin(9600);
}

void loop() {
  while(Serial.available()) {

    a= Serial.readString();// read the incoming data as string
    
    // set the cursor to column 0, line 1
    // (note: line 1 is the second row, since counting begins with 0):
    lcd.clear();
    lcd.setCursor(0, 0);
    // print the number of seconds since reset:
    
    lcd.print(a);
    
  }
}

