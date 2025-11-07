# GitHub Release Guide

## 🚀 Quick Release (Recommended)

### Method 1: Using Batch Script (Easiest)

1. Open Command Prompt (CMD)
2. Navigate to YanNian-Mol directory:
   ```cmd
   cd D:\FJTCM\DeepLife\DLProject\YanNian-Mol
   ```

3. Run the release script:
   ```cmd
   push_to_github.bat
   ```

4. Follow the prompts!

### Method 2: Manual Commands

Open Command Prompt in the YanNian-Mol directory and execute:

```cmd
# 1. Initialize Git
git init

# 2. Add all files
git add .

# 3. Create commit
git commit -m "Initial commit: YanNian-Mol v0.1.0"

# 4. Add remote repository
git remote add origin https://github.com/yuzhounaut/YanNian-Mol.git

# 5. Set main branch
git branch -M main

# 6. Push to GitHub
git push -u origin main

# 7. Create version tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## 🔑 Authentication

You'll need GitHub authentication when pushing:

### Using Personal Access Token (Recommended)

1. Visit: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Settings:
   - Note: `YanNian-Mol`
   - Expiration: Choose validity period
   - Check `repo` permission
4. Generate and copy the token
5. When pushing:
   - Username: Your GitHub username
   - Password: Paste the token (NOT your GitHub password!)

## ✅ Verify Release

After successful release, visit:
https://github.com/yuzhounaut/YanNian-Mol

You should see:
- ✓ All code files
- ✓ README displays correctly
- ✓ 86 files
- ✓ v0.1.0 tag

## 📝 Next Steps

### 1. Create Release (Optional but Recommended)

1. Visit: https://github.com/yuzhounaut/YanNian-Mol/releases/new
2. Select tag: v0.1.0
3. Release title: `v0.1.0 - Initial Release`
4. Description: Copy content from CHANGELOG.md
5. Click "Publish release"

### 2. Add Topics

1. Go to repository homepage
2. Click gear icon next to "About"
3. Add topics:
   - `deep-learning`
   - `pytorch`
   - `drug-discovery`
   - `bioinformatics`
   - `graph-neural-networks`
   - `molecular-modeling`
   - `longevity`
   - `aging-research`

### 3. Set Repository Description

In "About" settings, add:
```
AI-powered longevity prediction for model organisms | A deep learning framework for predicting compound effects on lifespan
```

### 4. Enable GitHub Pages (Optional)

To host documentation:
1. Settings → Pages
2. Source: Deploy from a branch
3. Branch: main, /docs
4. Save

## ❓ Common Issues

### Q: Push failed with authentication error
A: Use Personal Access Token instead of password

### Q: Push failed, repository doesn't exist
A: Confirm repository exists on GitHub: https://github.com/yuzhounaut/YanNian-Mol

### Q: Push failed with network error
A: Check network connection, or try using VPN

### Q: File too large to push
A: Check .gitignore correctly excludes large files (.pt, .npy, etc.)

### Q: Want to update already pushed code
A: 
```cmd
git add .
git commit -m "Update: describe changes"
git push
```

## 📞 Need Help?

If you encounter issues:
1. Check error messages
2. Verify network connection
3. Confirm GitHub repository is created
4. Ensure correct authentication method

---

**Good luck with your release! 🎉**
