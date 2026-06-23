import json
from kaggle_environments import make

def main():
    print("Loading replay.json...")
    with open("replay.json", "r") as f:
        replay_data = json.load(f)
        
    print("Initializing environment...")
    env = make("cabt")
    
    # Inject the loaded replay data into the environment object
    env.steps = replay_data.get("steps", [])
    if "configuration" in replay_data:
        # Update config but keep the environment's default keys if needed
        for k, v in replay_data["configuration"].items():
            env.configuration[k] = v
            
    print("Rendering to HTML...")
    html_out = env.render(mode="html")
    
    with open("replay.html", "w") as f:
        f.write(html_out)
        
    print("Successfully generated replay.html from the original replay.json!")

if __name__ == "__main__":
    main()
