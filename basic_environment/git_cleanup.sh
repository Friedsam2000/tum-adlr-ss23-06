#!/bin/bash

# Add 'venv' to .gitignore
echo 'venv/' >> .gitignore

# Add 'venv' to the staging area
git add venv/

# Clean the repository
git clean -df

# Unstage 'venv'
git reset venv/

echo "Cleanup completed successfully, 'venv' directory is preserved."

