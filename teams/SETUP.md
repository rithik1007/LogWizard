# Teams App Setup

## Quick Start

### 1. Replace placeholders in `manifest.json`
- `{{APP_ID}}` — Generate a GUID at https://www.guidgenerator.com/
- `{{BOT_ID}}` — From Azure Bot registration (step 3)
- `{{YOUR_DOMAIN}}` — Your hosted Echelon AI URL (e.g. `echelon.wkrainier.com`)

### 2. Add app icons
- `color.png` — 192x192 full-color icon
- `outline.png` — 32x32 transparent outline icon

### 3. Register a Bot (for Teams Bot feature)
1. Go to https://portal.azure.com → "Azure Bot" → Create
2. Choose **Multi Tenant** or **Single Tenant** based on your org
3. Note the **Bot ID** (Microsoft App ID) and **Password**
4. Set the messaging endpoint to: `https://{{YOUR_DOMAIN}}/api/teams/messages`
5. Add the Bot ID to your `.env` as `TEAMS_BOT_ID` and password as `TEAMS_BOT_PASSWORD`

### 4. Deploy Echelon AI
The server must be accessible over HTTPS on a public/internal URL.
Options:
- **Azure App Service** — `az webapp up --name echelon-ai --runtime PYTHON:3.9`
- **Azure Container Instance** — Build Docker image, deploy to ACI
- **Internal server** — Any server reachable from Teams with valid TLS

### 5. Package and sideload
```bash
cd teams/
zip -r echelon-teams.zip manifest.json color.png outline.png
```
Then in Teams: Apps → Manage your apps → Upload a custom app → Select the zip.

### How it works
- **Tab (sidebar)**: Loads the Echelon AI web UI inside Teams via iframe
- **Bot**: Users can chat with Echelon AI directly in Teams — messages go to
  `/api/teams/messages` which routes to the same agent
