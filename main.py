from flask import Flask, request
import simulation_engine
import threading

app = Flask(__name__)

def background_task(data):
    simulation_engine.run_simulation(data)

@app.route('/run_simulation', methods=['POST'])
def run():
    data = request.get_json()
    print("Received data:", data)

    thread = threading.Thread(target=background_task, args=(data,))
    thread.start()
        
    return 'Simulation is running'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)