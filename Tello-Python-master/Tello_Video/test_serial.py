import serial
from time import sleep
import sys

COM_PORT = 'COM3'
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)

try:
    while True:
        choice = input('按1開燈、按2關燈、按e關閉程式  ').lower()

        if choice == '1':
            print('傳送開燈指令')
            ser.write(b'1\n')  # 訊息必須是位元組類型
            sleep(0.5)              # 暫停0.5秒，再執行底下接收回應訊息的迴圈
        elif choice == '2':
            print('傳送關燈指令')
            ser.write(b'2\n')
            sleep(0.5)
        elif choice == '3':
            print('buzzer on')
            ser.write(b'3\n')
            sleep(0.5)
        elif choice == '4':
            print('buzzer off')
            ser.write(b'4\n')
            sleep(0.5)
        elif choice == 'e':
            ser.close()
            print('再見！')
            sys.exit()
        else:
            print('指令錯誤…')

        while ser.in_waiting:
            mcu_feedback = ser.readline().decode()  # 接收回應訊息並解碼
            print('控制板回應：', mcu_feedback)
            
except KeyboardInterrupt:
    ser.close()
    print('再見！')