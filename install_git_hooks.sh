#!/bin/bash
# Install git hooks for pipeline validation

set -e

echo "🔧 Installing pipeline validation git hooks..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository"
    exit 1
fi

# Create git hooks directory if it doesn't exist
mkdir -p .git/hooks

# Install pre-commit hook
if [ -f "git_hooks/pre-commit" ]; then
    cp git_hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo "✅ Installed pre-commit hook"
else
    echo "❌ pre-commit hook source not found"
    exit 1
fi

# Test the hook
echo "🧪 Testing git hook installation..."
if .git/hooks/pre-commit; then
    echo "✅ Git hooks installed and tested successfully"
else
    echo "❌ Git hook test failed"
    exit 1
fi

echo ""
echo "📝 Git hooks are now active. They will run automatically on commit."
echo "   To bypass the hook (not recommended): git commit --no-verify"
echo "   To update the pipeline index: python3 pipeline_autoscan.py"