# Deploying ragtrust to the VPS

One process (uvicorn, serving both the `/answer`/`/health`/`/config`/`/api/examples`
API and the static dashboard) behind one Caddy reverse-proxy block, using
OmniRoute (already running on the VPS at `http://127.0.0.1:20128`) as the LLM.

Run these by hand, on the VPS, logged in as `admin`.

## 1. Get the code and dependencies onto the VPS

```bash
mkdir -p /home/admin/projects
cd /home/admin/projects
git clone <this repo's URL> rag-trustworthiness-industrial
cd rag-trustworthiness-industrial

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

## 2. Confirm OmniRoute is reachable

```bash
curl -s http://127.0.0.1:20128/v1/models | head
```

Should list models including `auto/best-fast`. If this fails, stop here --
nothing downstream will work without it.

## 3. Build (or copy over) an index

Either build one on the VPS from a corpus you've copied there:

```bash
.venv/bin/ragtrust index /path/to/corpus --out /home/admin/projects/rag-trustworthiness-industrial/data/index
```

or `scp` a previously-built index directory to that same path from your
machine. Either way, `RAGTRUST_INDEX` (next step) must point at it.

## 4. Configure environment

```bash
cp deploy/.env.example .env
```

Edit `.env` and fill in the real `RAGTRUST_LLM_API_KEY` (the OmniRoute key)
and confirm `RAGTRUST_INDEX` points at the index directory from step 3.
`RAGTRUST_LLM_BASE_URL`, `RAGTRUST_LLM_MODEL`, and `RAGTRUST_GENERATOR` can be
left at the example's defaults unless you want a different OmniRoute model.

`.env` holds a live credential -- confirm it is not tracked by git
(`git check-ignore .env` should print `.env`; the repo's `.gitignore` already
excludes it) and leave its permissions readable only by `admin`
(`chmod 600 .env`).

## 5. Drop in the dashboard build (built separately, not part of this repo change)

Whoever built `dashboard/index.html` (+ assets) should place that directory at
`/home/admin/projects/rag-trustworthiness-industrial/dashboard/`. If it isn't
there yet, that's fine -- the service starts without it and serves the API
only (a warning is logged); add the directory later and restart the service
(step 7) to pick it up.

## 6. Install and start the systemd service

```bash
sudo cp deploy/ragtrust.service /etc/systemd/system/ragtrust.service
sudo systemctl daemon-reload
sudo systemctl enable --now ragtrust
sudo systemctl status ragtrust
```

Check the logs if it doesn't come up clean:

```bash
journalctl -u ragtrust -f
```

Smoke-test locally on the VPS (the service binds loopback-only, so this must
run on the VPS itself, not your machine):

```bash
curl -s http://127.0.0.1:8402/health
curl -s -X POST http://127.0.0.1:8402/answer \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is this corpus about?"}'
```

## 7. Wire up Caddy

Append the site block (do not overwrite the existing Caddyfile):

```bash
cat deploy/Caddyfile.snippet | sudo tee -a /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy auto-issues a TLS cert for `ragtrust.169.58.185.43.sslip.io` on reload
(sslip.io resolves that hostname to the VPS's own IP, so no separate DNS
records are needed).

## 8. Verify end-to-end

From your own machine:

```bash
curl -s https://ragtrust.169.58.185.43.sslip.io/health
```

and open `https://ragtrust.169.58.185.43.sslip.io/` in a browser -- it should
serve `dashboard/index.html` if step 5's directory is in place, or a 404 (API
still reachable at `/health`, `/config`, `/answer`, `/api/examples`) if not.

## Updating after a code change

```bash
cd /home/admin/projects/rag-trustworthiness-industrial
git pull
.venv/bin/pip install -e .        # only if dependencies changed
sudo systemctl restart ragtrust
```

## Rolling back

```bash
cd /home/admin/projects/rag-trustworthiness-industrial
git log --oneline -5      # find the commit to roll back to
git checkout <commit>
sudo systemctl restart ragtrust
```
