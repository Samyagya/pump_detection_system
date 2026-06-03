# 🤝 Team Collaboration Guide
**Project:** Pulse-Driven Supply Chain Agent
**Repository:** `tableau-agent`

---

## 🚨 The Golden Rule
> **NEVER push directly to the `main` branch.**

The `main` branch is our "Production" code. It must always be working. If you push broken code to `main`, you block the entire team from working. Always use a **new branch** for your changes.

---

## 🛠️ Setup (One-Time Only)
Before writing any code, ensure your local environment is safe.

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tableau-agent.git

2. **Check the Safety Net:** Ensure the file `.gitignore` exists in the root folder. This prevents you from accidentally uploading passwords, virtual environments (`venv`), or junk files.

---

## 🔄 Daily Workflow (The 5 Steps)

Every time you want to write code, follow this exact cycle.

### Step 1: Sync (Start Fresh)
Before you start coding, get the latest changes from the team.

```bash
git checkout main
git pull origin main
```

### Step 2: Create a Safe Space (Branch)
Create a new branch for your specific task. Name it clearly (e.g., `feature-slack-bot`, `fix-csv-typo`).

```bash
git checkout -b feature-your-task-name
```

### Step 3: Code & Commit
Do your work. When you are ready to save:

```bash
git add .
git commit -m "Description of what you did"
```

### Step 4: Push to Github
Send your branch to the cloud (NOT to main).

```bash
git push origin feature-your-task-name
```

### Step 5: The Pull Request (PR)
1. Go to the GitHub repository in your browser.
2. Click **"Compare & pull request"** on the yellow banner that appears.
3. Click **Create Pull Request**.
4. **Do NOT merge it yourself.** Send the link to the group chat: *"Can someone review my PR?"*
5. Once a teammate approves, click **Merge**.

---

### 🛑 Emergency Guide

**"I tried to push and got an error!"**

* **Cause:** You probably forgot to `pull` before you started, or you are trying to push to `main` directly.
* **Fix:** Ensure you are on your feature branch (`git checkout feature-name`) and try pushing again.

**"I have a Merge Conflict!"**

* **Don't Panic.** This means two people edited the same line of code.
* **Fix:** GitHub will disable the "Merge" button. You must fix the conflict in the GitHub web editor or locally in VS Code before you can finish. Ask the team for help if stuck!
