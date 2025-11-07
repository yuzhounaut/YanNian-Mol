@echo off
REM YanNian-Mol GitHub Release Script

echo ================================================================================
echo YanNian-Mol GitHub Release Script
echo ================================================================================
echo.

REM Check if in correct directory
if not exist "lifespan_predictor" (
    echo Error: Please run this script in the YanNian-Mol directory
    pause
    exit /b 1
)

echo Step 1: Initialize Git repository
git init
if errorlevel 1 (
    echo Git initialization failed
    pause
    exit /b 1
)
echo [OK] Git repository initialized
echo.

echo Step 2: Add all files
git add .
if errorlevel 1 (
    echo Failed to add files
    pause
    exit /b 1
)
echo [OK] Files added
echo.

echo Step 3: Create initial commit
git commit -m "Initial commit: YanNian-Mol v0.1.0 - AI-powered longevity prediction for model organisms"
if errorlevel 1 (
    echo Commit failed
    pause
    exit /b 1
)
echo [OK] Commit created
echo.

echo Step 4: Add remote repository
git remote add origin https://github.com/yuzhounaut/YanNian-Mol.git
if errorlevel 1 (
    echo Note: Remote repository may already exist
)
echo [OK] Remote repository added
echo.

echo Step 5: Set main branch
git branch -M main
echo [OK] Main branch set
echo.

echo Step 6: Push to GitHub
echo.
echo IMPORTANT: You may need to enter GitHub credentials
echo Recommend using Personal Access Token instead of password
echo Generate token at: https://github.com/settings/tokens
echo.
pause

git push -u origin main
if errorlevel 1 (
    echo.
    echo [FAILED] Push failed!
    echo.
    echo Possible reasons:
    echo 1. Repository doesn't exist - Please create it on GitHub first
    echo 2. Authentication failed - Please use Personal Access Token
    echo 3. Network issue - Please check your connection
    echo.
    pause
    exit /b 1
)
echo.
echo [OK] Code pushed to GitHub
echo.

echo Step 7: Create version tag
git tag -a v0.1.0 -m "Release v0.1.0 - Initial release"
git push origin v0.1.0
if errorlevel 1 (
    echo Tag push failed (may already exist)
) else (
    echo [OK] Version tag created
)
echo.

echo ================================================================================
echo SUCCESS! 
echo ================================================================================
echo.
echo [OK] Your code has been successfully pushed to GitHub
echo.
echo Repository: https://github.com/yuzhounaut/YanNian-Mol
echo.
echo Next steps:
echo 1. Visit repository to view code
echo 2. Create Release (optional)
echo    https://github.com/yuzhounaut/YanNian-Mol/releases/new
echo 3. Add Topics (optional)
echo    Suggested: deep-learning, pytorch, drug-discovery, bioinformatics
echo.
echo ================================================================================
pause
