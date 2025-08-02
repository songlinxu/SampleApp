from flask import Flask, request, jsonify, render_template
from rl_agent import TrainedRL

app = Flask(__name__)
agent = TrainedRL()

# Map buttons A-D to 4D CartPole state vectors
state_map = {
    "A": [0.0, 0.0, 0.0, 0.0],       # balanced state
    "B": [0.05, 0.1, 0.02, -0.01],  # slight tilt
    "C": [-0.1, -0.2, 0.15, 0.1],   # more tilt
    "D": [0.2, 0.3, -0.2, -0.3]     # unstable
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/interact", methods=["POST"])
def interact():
    try:
        button = request.json.get("button")
        state = state_map.get(button, [0.0, 0.0, 0.0, 0.0])
        agent.update_state(state)
        action = agent.update_action()
        return jsonify({"action": action})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
