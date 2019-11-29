char userInput;
int relayPin = 8;

void setup() {
  Serial.begin(9600);
  pinMode(relayPin, OUTPUT);
}

void loop() {
  if(Serial.available()>0){
    userInput = Serial.read();
    if(userInput == 'g'){
      digitalWrite(relayPin, HIGH); // g for go
    }
    else if (userInput == 's'){
      digitalWrite(relayPin, LOW); // s for stop
    }
  }
}
