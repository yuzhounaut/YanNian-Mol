#!/usr/bin/env python
"""
Script to help with GitHub release process.

This script guides you through:
1. Initializing git repository
2. Creating initial commit
3. Connecting to GitHub
4. Pushing code
5. Creating release tag
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture=True):
    """Run a shell command."""
    print(f"\n→ Running: {cmd}")
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
    else:
        result = subprocess.run(cmd, shell=True)

    if check and result.returncode != 0:
        print(f"\n✗ Command failed with exit code {result.returncode}")
        return False
    return True


def check_git_installed():
    """Check if git is installed."""
    result = subprocess.run("git --version", shell=True, capture_output=True)
    return result.returncode == 0


def is_git_repo():
    """Check if current directory is a git repository."""
    result = subprocess.run("git status", shell=True, capture_output=True)
    return result.returncode == 0


def get_user_input(prompt, default=None):
    """Get user input with optional default."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    response = input(prompt).strip()
    return response if response else default


def main():
    """Main function to guide GitHub release."""
    print("=" * 80)
    print("GitHub Release Helper for lifespan_predictor")
    print("=" * 80)

    # Check git installation
    if not check_git_installed():
        print("\n✗ Git is not installed!")
        print("Please install Git from: https://git-scm.com/downloads")
        sys.exit(1)

    print("\n✓ Git is installed")

    # Check if already a git repo
    if is_git_repo():
        print("\n✓ Git repository already initialized")
        print("\nCurrent git status:")
        run_command("git status", check=False)

        response = get_user_input(
            "\nDo you want to continue with existing repository? (y/n)", "y"
        )
        if response.lower() != "y":
            print("Exiting...")
            sys.exit(0)
    else:
        print("\n→ Initializing git repository...")
        if not run_command("git init"):
            sys.exit(1)
        print("✓ Git repository initialized")

    # Get GitHub username
    print("\n" + "=" * 80)
    print("GitHub Repository Setup")
    print("=" * 80)

    github_username = get_user_input("\nEnter your GitHub username")
    if not github_username:
        print("✗ GitHub username is required")
        sys.exit(1)

    repo_name = get_user_input(
        "Enter repository name", "lifespan-predictor"
    )

    # Confirm repository details
    print(f"\n→ Repository will be: https://github.com/{github_username}/{repo_name}")
    response = get_user_input("Is this correct? (y/n)", "y")
    if response.lower() != "y":
        print("Please run the script again with correct details")
        sys.exit(0)

    # Check if remote already exists
    result = subprocess.run(
        "git remote get-url origin", shell=True, capture_output=True
    )
    if result.returncode == 0:
        current_remote = result.stdout.decode().strip()
        print(f"\n→ Remote 'origin' already exists: {current_remote}")
        response = get_user_input("Do you want to update it? (y/n)", "n")
        if response.lower() == "y":
            run_command(
                f"git remote set-url origin https://github.com/{github_username}/{repo_name}.git"
            )
    else:
        print("\n→ Adding GitHub remote...")
        run_command(
            f"git remote add origin https://github.com/{github_username}/{repo_name}.git"
        )

    # Stage and commit files
    print("\n" + "=" * 80)
    print("Committing Files")
    print("=" * 80)

    print("\n→ Checking for uncommitted changes...")
    result = subprocess.run(
        "git status --porcelain", shell=True, capture_output=True, text=True
    )

    if result.stdout.strip():
        print("\n→ Staging all files...")
        run_command("git add .")

        print("\n→ Creating commit...")
        commit_message = """Initial commit: lifespan_predictor v0.1.0

- Refactored notebook-based code into modular package
- Added comprehensive documentation
- Implemented testing infrastructure
- Added CI/CD configuration
- Created validation framework
- Prepared release package v0.1.0"""

        # Use a temporary file for the commit message to handle multiline
        with open(".git_commit_msg.tmp", "w") as f:
            f.write(commit_message)

        run_command('git commit -F .git_commit_msg.tmp')
        os.remove(".git_commit_msg.tmp")
        print("✓ Files committed")
    else:
        print("✓ No uncommitted changes")

    # Push to GitHub
    print("\n" + "=" * 80)
    print("Pushing to GitHub")
    print("=" * 80)

    print("\n⚠ Important: Make sure you have created the repository on GitHub!")
    print(f"   Go to: https://github.com/new")
    print(f"   Repository name: {repo_name}")
    print(f"   Do NOT initialize with README, .gitignore, or license")

    response = get_user_input("\nHave you created the repository on GitHub? (y/n)", "n")
    if response.lower() != "y":
        print("\nPlease create the repository on GitHub first, then run this script again")
        print(f"URL: https://github.com/new")
        sys.exit(0)

    print("\n→ Pushing to GitHub...")
    print("   (You may be prompted for GitHub credentials)")

    # Set main branch
    run_command("git branch -M main", check=False)

    # Push
    if not run_command("git push -u origin main", check=False):
        print("\n✗ Push failed!")
        print("\nTroubleshooting:")
        print("1. Make sure the repository exists on GitHub")
        print("2. Check your GitHub credentials")
        print("3. You may need to use a Personal Access Token instead of password")
        print("   Generate one at: https://github.com/settings/tokens")
        print("\nTry pushing manually:")
        print(f"   git push -u origin main")
        sys.exit(1)

    print("\n✓ Code pushed to GitHub!")

    # Create release tag
    print("\n" + "=" * 80)
    print("Creating Release Tag")
    print("=" * 80)

    version = Path("VERSION").read_text().strip()
    print(f"\n→ Creating tag v{version}...")

    run_command(f'git tag -a v{version} -m "Release v{version} - Initial Release"')
    run_command(f"git push origin v{version}")

    print(f"\n✓ Tag v{version} created and pushed!")

    # Final instructions
    print("\n" + "=" * 80)
    print("SUCCESS! 🎉")
    print("=" * 80)

    print(f"\n✓ Your code is now on GitHub!")
    print(f"   Repository: https://github.com/{github_username}/{repo_name}")
    print(f"   Tag: v{version}")

    print("\n→ Next steps:")
    print(f"   1. Visit: https://github.com/{github_username}/{repo_name}/releases")
    print(f"   2. Click 'Create a new release'")
    print(f"   3. Select tag: v{version}")
    print(f"   4. Copy release notes from CHANGELOG.md")
    print(f"   5. Publish release")

    print("\n→ Optional: Build and attach distribution files")
    print("   python -m build")
    print("   Then attach files from dist/ folder to the release")

    print("\n→ Share your project:")
    print(f"   Installation: pip install git+https://github.com/{github_username}/{repo_name}.git")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        sys.exit(1)
