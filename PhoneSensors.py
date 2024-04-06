import websocket
import json
import threading

# Global variable to store orientation
orientation = {'x': 0, 'y': 0, 'z': 0}

def on_message(ws, message):
    global orientation
    values = json.loads(message)['values']
    orientation = {'x': values[0], 'y': values[1], 'z': values[2]}

def on_error(ws, error):
    print("error occurred ", error)

def on_close(ws, close_code, reason):
    print("connection closed : ", reason)

def on_open(ws):
    print("connected")

def connect(url):
    url = f'{url}/sensor/connect?type=android.sensor.accelerometer'

    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()

def start_sensor_thread(url):
    sensor_thread = threading.Thread(target=connect, args=(url,))
    sensor_thread.start()

if __name__ == '__main__':
    phone_url = 'ws://pixel-8-pro.lan:8080'
    start_sensor_thread(phone_url)