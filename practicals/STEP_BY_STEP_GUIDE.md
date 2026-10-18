# PetalWise — step-by-step practical guide

This guide adapts the supplied 2CEAI702 MLOps practical file to PetalWise. Follow the official manual's screenshot placeholders and sequence.

## Practical 1 — Flask ML system

1. Extract this folder and open it in VS Code.
2. Open terminal in this directory.
3. Check Python: `python --version`. Prefer Python 3.10 or 3.11 per manual.
4. Create environment: `python -m venv .venv`
5. Install dependencies using `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`
6. Train/serialize: `.\.venv\Scripts\python.exe train_model.py`. Confirm `models/model.pkl` and `metrics.json` appear.
7. Run Flask: `.\.venv\Scripts\python.exe app.py`
8. Visit `http://127.0.0.1:5000`.
9. Check health: `Invoke-RestMethod http://127.0.0.1:5000/health`.
10. Get API info: `Invoke-RestMethod http://127.0.0.1:5000/prediction`.
11. Test JSON prediction with:
```powershell
$body = @{sepal_length=5.1; sepal_width=3.5; petal_length=1.4; petal_width=0.2} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/prediction -ContentType "application/json" -Body $body
```
12. Capture real screenshots for environment, training output/model, project structure, web form, prediction, health and API response.

## Practical 2 — Docker

1. Install Docker Desktop and start it.
2. Run `docker --version` and `docker info`.
3. Build: `docker build -t petalwise-iris:1.0 .`
4. List image: `docker images`.
5. Run: `docker run -d --name petalwise-iris-app -p 5000:5000 petalwise-iris:1.0`
6. Check: `docker ps`; logs: `docker logs petalwise-iris-app`.
7. Open `http://localhost:5000`, test prediction and `/health`.
8. Stop/remove: `docker stop petalwise-iris-app`; `docker rm petalwise-iris-app`.
9. Capture genuine screenshots of build, images, running container, logs and browser.

## Practical 3 — AWS EC2

1. Sign in to AWS Console and choose a region (the manual gives Mumbai `ap-south-1` as an example).
2. Launch an Ubuntu LTS EC2 instance. Store the SSH private key securely.
3. Configure inbound rules: SSH 22 from your IP only; TCP 5000 from a trusted source range or public only if the assignment requires it.
4. SSH into the instance.
5. Install Docker if needed: `sudo apt update && sudo apt install -y docker.io`; then `sudo systemctl enable --now docker`.
6. Transfer the project folder to the VM using `scp` or upload a ZIP securely.
7. From the project folder on EC2, build: `sudo docker build -t petalwise-iris:1.0 .`
8. Run: `sudo docker run -d --name petalwise-iris-app --restart unless-stopped -p 5000:5000 petalwise-iris:1.0`.
9. Check `sudo docker ps`, `sudo docker logs petalwise-iris-app`.
10. Test `http://YOUR_EC2_PUBLIC_IP:5000` and `/health`.
11. Capture real screenshots. Stop/terminate resources when no longer needed to avoid charges.

## Practical 4 — Google Cloud Compute Engine

1. Create/select a GCP project and link billing only if required; check costs.
2. Enable Compute Engine API.
3. Create an Ubuntu LTS VM.
4. Add a firewall ingress rule for TCP 5000 targeting only the intended VM and source range.
5. Use browser SSH to connect.
6. Install Docker: `sudo apt update && sudo apt install -y docker.io`; `sudo systemctl enable --now docker`.
7. Transfer project files to VM using `gcloud compute scp` or a secure alternative.
8. Build and run using the same Docker commands as Practical 3.
9. Test `http://YOUR_GCP_EXTERNAL_IP:5000` and `/health`.
10. Capture real screenshots and stop/delete resources when finished.

## GitHub and README

1. Create a repository named `petalwise-iris-mlops`.
2. Upload code, templates, static images, tests, Dockerfile, requirements and README. Do not upload `.venv`, private keys or secrets.
3. Add actual tested AWS and GCP URLs to README.
4. Add the GitHub repository URL.
5. Complete every practical screenshot placeholder in the official DOCX and submit through the instructor's form.

Never fabricate screenshots, cloud URLs, accuracy results or completed deployment evidence.
