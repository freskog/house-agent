#!/bin/bash
# Simple setup script for speech recognition components

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up speech recognition components...${NC}"

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script is intended for macOS only."
    exit 1
fi

# Check for Apple Silicon
if [[ "$(uname -m)" == "arm64" ]]; then
    IS_APPLE_SILICON=true
    echo "✅ Detected Apple Silicon Mac"
else
    IS_APPLE_SILICON=false
    echo "ℹ️ Running on Intel Mac"
fi

# Check if Homebrew is installed, install if not
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install required packages
echo "Installing required packages..."
brew install portaudio ffmpeg

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install pywhispercpp with CoreML support
echo "Installing pywhispercpp with CoreML support..."
pip uninstall -y pywhispercpp
if [ "$IS_APPLE_SILICON" = true ]; then
    echo "Installing pywhispercpp with CoreML support for Apple Silicon..."
    WHISPER_COREML=1 pip install pywhispercpp
    
    # Set CoreML environment variable permanently in shell profiles
    echo "Setting up permanent CoreML environment variable..."
    
    # Determine which shell the user is using
    SHELL_TYPE=$(basename "$SHELL")
    
    if [ "$SHELL_TYPE" = "zsh" ]; then
        # For zsh
        if ! grep -q "export WHISPER_COREML=1" ~/.zshrc; then
            echo "export WHISPER_COREML=1" >> ~/.zshrc
            echo "Added WHISPER_COREML to ~/.zshrc"
        fi
    elif [ "$SHELL_TYPE" = "bash" ]; then
        # For bash
        if ! grep -q "export WHISPER_COREML=1" ~/.bashrc; then
            echo "export WHISPER_COREML=1" >> ~/.bashrc
            echo "Added WHISPER_COREML to ~/.bashrc"
        fi
        if ! grep -q "export WHISPER_COREML=1" ~/.bash_profile; then
            echo "export WHISPER_COREML=1" >> ~/.bash_profile
            echo "Added WHISPER_COREML to ~/.bash_profile"
        fi
    fi
    
    # Set for current session
    export WHISPER_COREML=1
    echo "CoreML support enabled for current session"
else
    echo "Installing pywhispercpp for Intel Mac..."
    pip install pywhispercpp
fi

# Check if virtual environment is active and add to its activate script
if [ -n "$VIRTUAL_ENV" ]; then
    ACTIVATE_SCRIPT="$VIRTUAL_ENV/bin/activate"
    if [ -f "$ACTIVATE_SCRIPT" ]; then
        if ! grep -q "export WHISPER_COREML=1" "$ACTIVATE_SCRIPT"; then
            echo "export WHISPER_COREML=1" >> "$ACTIVATE_SCRIPT"
            echo "Added WHISPER_COREML to virtual environment activate script"
        fi
    fi
fi

# Create .env file for Python applications to read
echo "WHISPER_COREML=1" > .env
echo "Created .env file with WHISPER_COREML=1"

# Verify installation
echo "Verifying installation..."
python -c "import pyaudio; print('PyAudio installed successfully')"
python -c "import pywhispercpp; from pywhispercpp.model import Model; print('pywhispercpp installed successfully')"

echo "Setup complete! You can now run: python speech_test.py"
echo -e "${YELLOW}Important: You may need to restart your terminal for CoreML support to take effect.${NC}" 