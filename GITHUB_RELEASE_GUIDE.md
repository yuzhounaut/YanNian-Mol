# GitHub Release Guide

This guide will walk you through releasing the lifespan_predictor project to GitHub.

## Prerequisites

- GitHub account (you have this ✓)
- Git installed on your system
- Project files ready (you have this ✓)

## Step 1: Initialize Git Repository

Open your terminal in the project directory and run:

```bash
# Initialize git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: lifespan_predictor v0.1.0

- Refactored notebook-based code into modular package
- Added comprehensive documentation
- Implemented testing infrastructure
- Added CI/CD configuration
- Created validation framework
- Prepared release package v0.1.0"
```

## Step 2: Create GitHub Repository

### Option A: Using GitHub Web Interface (Recommended)

1. Go to https://github.com/new
2. Fill in the repository details:
   - **Repository name**: `lifespan-predictor` (or your preferred name)
   - **Description**: "A modular deep learning pipeline for predicting compound effects on C. elegans lifespan"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```bash
gh repo create lifespan-predictor --public --source=. --remote=origin --description="A modular deep learning pipeline for predicting compound effects on C. elegans lifespan"
```

## Step 3: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Run these commands:

```bash
# Add GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/lifespan-predictor.git

# Verify remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

**Replace `YOUR_USERNAME` with your actual GitHub username!**

## Step 4: Create a Release on GitHub

### Option A: Using GitHub Web Interface

1. Go to your repository on GitHub
2. Click on "Releases" (right sidebar)
3. Click "Create a new release"
4. Fill in the release details:
   - **Tag version**: `v0.1.0`
   - **Release title**: `v0.1.0 - Initial Release`
   - **Description**: Copy content from CHANGELOG.md for v0.1.0
5. Optionally attach distribution files:
   - Build the package: `python -m build`
   - Attach files from `dist/` folder
6. Click "Publish release"

### Option B: Using Git Tags and GitHub CLI

```bash
# Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0 - Initial Release"
git push origin v0.1.0

# Create GitHub release (if gh CLI is installed)
gh release create v0.1.0 --title "v0.1.0 - Initial Release" --notes-file CHANGELOG.md
```

## Step 5: Add Repository Badges (Optional but Recommended)

Add these badges to the top of your README.md:

```markdown
# Lifespan Predictor

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub release](https://img.shields.io/github/v/release/YOUR_USERNAME/lifespan-predictor.svg)](https://github.com/YOUR_USERNAME/lifespan-predictor/releases)
```

## Step 6: Set Up GitHub Actions (CI/CD)

Your project already has `.github/workflows/ci.yml`. After pushing to GitHub:

1. Go to your repository
2. Click on "Actions" tab
3. You should see the CI workflow
4. It will run automatically on push and pull requests

## Step 7: Configure Repository Settings (Optional)

### Add Topics
1. Go to repository main page
2. Click the gear icon next to "About"
3. Add topics: `deep-learning`, `pytorch`, `drug-discovery`, `bioinformatics`, `graph-neural-networks`, `molecular-modeling`

### Enable Issues and Discussions
1. Go to Settings → General
2. Enable "Issues" for bug tracking
3. Enable "Discussions" for community Q&A

### Set Up Branch Protection (Recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - Require pull request reviews
   - Require status checks to pass (CI tests)
   - Require branches to be up to date

## Quick Command Reference

Here's a complete script you can run:

```bash
# 1. Initialize and commit
git init
git add .
git commit -m "Initial commit: lifespan_predictor v0.1.0"

# 2. Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/lifespan-predictor.git

# 3. Push to GitHub
git branch -M main
git push -u origin main

# 4. Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0 - Initial Release"
git push origin v0.1.0

# 5. Build distribution (optional)
python -m build
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:

**Option 1: Use Personal Access Token (Recommended)**
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing

**Option 2: Use SSH**
```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to GitHub
# Copy the public key: cat ~/.ssh/id_ed25519.pub
# Add it at: GitHub Settings → SSH and GPG keys

# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/lifespan-predictor.git
```

### Large Files

If you have large data files (>100MB):
1. Add them to `.gitignore`
2. Or use Git LFS:
```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.npy"
git add .gitattributes
```

### Commit History

If you want to clean up commit history before pushing:
```bash
# Squash all commits into one
git reset $(git commit-tree HEAD^{tree} -m "Initial commit: lifespan_predictor v0.1.0")
```

## After Release

### Update README
Add installation instructions:
```markdown
## Installation

### From GitHub
\`\`\`bash
pip install git+https://github.com/YOUR_USERNAME/lifespan-predictor.git
\`\`\`

### From Source
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/lifespan-predictor.git
cd lifespan-predictor
pip install -e .
\`\`\`
```

### Share Your Project
- Tweet about it
- Post on Reddit (r/MachineLearning, r/bioinformatics)
- Share on LinkedIn
- Add to awesome lists

## Next Steps

1. ✓ Push code to GitHub
2. ✓ Create v0.1.0 release
3. Consider publishing to PyPI (see PYPI_RELEASE_GUIDE.md)
4. Set up documentation hosting (GitHub Pages or Read the Docs)
5. Add contributing guidelines (CONTRIBUTING.md)
6. Create issue templates
7. Add code of conduct

## Need Help?

- GitHub Docs: https://docs.github.com
- Git Basics: https://git-scm.com/book/en/v2
- GitHub CLI: https://cli.github.com/

---

**Remember to replace `YOUR_USERNAME` with your actual GitHub username in all commands!**
