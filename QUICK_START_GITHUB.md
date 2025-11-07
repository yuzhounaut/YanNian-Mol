# Quick Start: Release to GitHub

## Option 1: Automated Script (Easiest)

Run the automated helper script:

```bash
python scripts/github_release.py
```

The script will guide you through:
- Initializing git repository
- Creating initial commit
- Connecting to GitHub
- Pushing code
- Creating release tag

## Option 2: Manual Steps (5 minutes)

### Step 1: Create Repository on GitHub
1. Go to https://github.com/new
2. Repository name: `lifespan-predictor`
3. Description: "A modular deep learning pipeline for predicting compound effects on C. elegans lifespan"
4. Choose Public or Private
5. **DO NOT** check any initialization options
6. Click "Create repository"

### Step 2: Push Your Code

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: lifespan_predictor v0.1.0"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/lifespan-predictor.git

# Push
git branch -M main
git push -u origin main
```

### Step 3: Create Release

```bash
# Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

Then go to: https://github.com/YOUR_USERNAME/lifespan-predictor/releases
- Click "Create a new release"
- Select tag v0.1.0
- Copy release notes from CHANGELOG.md
- Publish!

## Authentication

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your password!)
  - Generate at: https://github.com/settings/tokens
  - Select scope: `repo`

## That's It!

Your project is now on GitHub! 🎉

Share it:
```bash
pip install git+https://github.com/YOUR_USERNAME/lifespan-predictor.git
```

## Need Help?

See detailed guide: `GITHUB_RELEASE_GUIDE.md`
