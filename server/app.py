from fastapi import FastAPI
from openenv.core.env_server import create_web_interface_app
from travel_pro.server.env import TravelEnv
from travel_pro.models import TravelAction, TravelObservation

# Initialize the OpenEnv server with the web interface enabled
# This helper automatically creates the FastAPI app and mounts the Gradio UI at /web
app = create_web_interface_app(
    env=TravelEnv,
    action_cls=TravelAction,
    observation_cls=TravelObservation,
    env_name="travel_pro"
)

# Optional: You can still add custom routes here if needed, 
# but create_web_interface_app already handles / and /web redirects.

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
