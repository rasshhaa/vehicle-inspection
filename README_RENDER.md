Deploying to Render

Overview
- This repository contains a FastAPI backend that also serves the frontend static files.
- The Render web service will run `uvicorn backend.main:app` and expose the app.

Files added for Render
- `requirements.txt` (root): Python dependencies.
- `Procfile`: start command used by Render.

Steps to deploy
1. Push this repo to GitHub/GitLab.
2. Create a new Web Service on Render (https://render.com/dashboard).
   - Connect your Git repository.
   - For "Environment", choose "Python".
   - Build Command: leave empty (Render will run `pip install -r requirements.txt` automatically), or set: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Set the root directory to the repository root.
3. Add environment variables if needed (e.g., Roboflow API key). In production you should NOT hardcode API keys.

Notes & Post-deploy checks
- Ensure `uploads/` and `static/` directories are writeable; Render's ephemeral filesystem is available during runtime but will not persist between deploys. For persistent uploads consider using S3 and update `UPLOAD_DIR` accordingly.
- Verify the frontend loads at `/` and API endpoints like `/inspect` and `/report` work.
- If you experience camera issues from iVCam, ensure the server is reachable from your machine and the browser can access camera stream.

Advanced: render.yaml
- If you want Render to auto-create the service from `render.yaml`, I can generate a sample `render.yaml` for you.
