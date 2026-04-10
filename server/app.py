from fastapi import FastAPI
from openenv.core.env_server import HTTPEnvServer
from travel_pro.server.env import TravelEnv
from travel_pro.models import TravelAction, TravelObservation

# Create the FastAPI application
app = FastAPI(title="Travel Pro Environment Server")

# Initialize the OpenEnv server with our environment class and models
env_server = HTTPEnvServer(
    env=TravelEnv,
    action_cls=TravelAction,
    observation_cls=TravelObservation
)

# Register the routes on the app
env_server.register_routes(app)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
