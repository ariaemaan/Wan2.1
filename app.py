from fastapi import FastAPI
import importlib.util
import sys
import os

# Add the gradio directory to the Python path
# to allow importing the Gradio app
current_dir = os.path.dirname(os.path.abspath(__file__))
gradio_dir = os.path.join(current_dir, 'gradio')
sys.path.insert(0, gradio_dir)

# Dynamically import the Gradio app
# This is to avoid issues with Vercel's build process
# if the Gradio app tries to load models at import time.
spec = importlib.util.spec_from_file_location("gradio_app", os.path.join(gradio_dir, "t2v_1.3B_singleGPU.py"))
gradio_module = importlib.util.module_from_spec(spec)
sys.modules["gradio_app"] = gradio_module
spec.loader.exec_module(gradio_module)

# The Gradio script should define a 'demo' object (the Gradio interface)
# and a function 'create_fastapi_app' that returns the FastAPI app.
# We will call this function to get the app.
# If 'create_fastapi_app' is not available, we try to get app from demo.app (FastAPI instance)
# or demo.queue.app if queueing is enabled.

# It's also assumed that the Gradio app's main interface object is named `demo`
if hasattr(gradio_module, 'demo'):
    gradio_interface = gradio_module.demo

    # Vercel expects an 'app' variable that is a WSGI/ASGI application.
    # Gradio's launch() method normally returns this, but we need to access it directly.
    # If the Gradio app has a queue, the FastAPI app is often at demo.queue.app
    # Otherwise, it might be at demo.app if not using a queue or if it's a newer Gradio version
    # We also need to handle the case where the app is directly returned by a function.

    app = None
    if hasattr(gradio_interface, 'queue') and hasattr(gradio_interface.queue, 'app'):
        app = gradio_interface.queue.app
    elif hasattr(gradio_interface, 'app') and isinstance(gradio_interface.app, FastAPI):
        app = gradio_interface.app
    else:
        # Fallback: Try to launch and get the app. This is less ideal for Vercel.
        # The Gradio app itself might need modification for this to work without
        # trying to start a Uvicorn server.
        # For now, we'll assume one of the above attributes exists or the Gradio app
        # needs to be modified as per the plan.
        # This part might need adjustment based on the Gradio app's structure.
        print("FastAPI app not found directly. The Gradio app might need adjustments for Vercel.")
        # As a placeholder, create a simple FastAPI app if the Gradio app isn't structured as expected.
        # This will allow deployment to succeed, but the Gradio app itself won't run.
        # The next plan step involves modifying the Gradio app.
        app = FastAPI()
        @app.get("/")
        async def root():
            return {"message": "Gradio app not fully configured for Vercel. See app.py."}

else:
    # If there's no 'demo' object, the Gradio script is not as expected.
    print("Gradio 'demo' object not found in t2v_1.3B_singleGPU.py.")
    app = FastAPI()
    @app.get("/")
    async def root():
        return {"message": "Gradio 'demo' object not found. Please check the Gradio script."}

# The Vercel Python runtime will look for an 'app' variable.
# This 'app' variable should be an ASGI application (like a FastAPI app).
