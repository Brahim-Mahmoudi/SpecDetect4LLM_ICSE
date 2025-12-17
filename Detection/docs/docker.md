![Overview](/static/web-app.png)

#  Running SpecDetect4LLM Web App with Docker

This guide explains how to containerize and run the SpecDetect4LLM web application using Docker. All you need is Docker installed on your machine.

---

##  Step 1 — Prepare the Docker Environment

Ensure you have the following files ready in the root directory of your project (`CODE_SMELL_LLM/`):

1. **`Dockerfile`**: (The instructions for building the image, created in the previous step).
2. **`requirements.txt`**: (The list of Python dependencies: `Flask`, etc.).
3. **`.dockerignore`**: (To exclude unnecessary files like `.git`, `venv`, etc.).
4. The application directories: `Detection/` and `web-app/`.

---

##  Step 2 — Build the Docker Image

Navigate to the root of your project (`CODE_SMELL_LLM/`) and run the build command:
```bash
docker build -t specdetect4llm-web .
```

This command reads your `Dockerfile` and creates a Docker image named `specdetect4llm-web`.

**Note**: The first build might take a few minutes as Docker downloads the base Python image and installs dependencies.

---

##  Step 3 — Run the Web Server Container

Once the image is built, run the container using the following command. We will map the internal port 5000 (where Flask runs) to an external port, like 8080, on your host machine.
```bash
docker run -d \
  -p 8080:5000 \
  --name specdetect4llm_app \
  specdetect4llm-web
```

| Option | Description |
|--------|-------------|
| `-d` | Runs the container in the background (detached mode). |
| `-p 8080:5000` | Maps your host machine's port 8080 to the container's exposed port 5000. |
| `--name ...` | Assigns an easy-to-use name to the running container instance. |
| `specdetect4llm-web` | The name of the image we built in Step 2. |

---

##  Step 4 — Access the Web Application

After running the container, the web application should be available in your browser at:
```
http://localhost:8080
```

You can now upload your project archives (ZIP/TAR) and run the analysis through the modern web interface.

---

##  Container Management (Useful Commands)

Here are a few useful Docker commands for managing your application:

| Command | Description |
|---------|-------------|
| `docker stop specdetect4llm_app` | Stops the running container instance. |
| `docker rm specdetect4llm_app` | Deletes the stopped container instance. |
| `docker rmi specdetect4llm-web` | Deletes the Docker image (if you need to rebuild it completely). |
| `docker logs specdetect4llm_app -f` | Displays the server logs in real-time for debugging purposes. |
