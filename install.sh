#!/bin/bash

# QuantsLab Installation Script
# This script sets up the complete development environment for QuantsLab

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect user's shell for conda activation
detect_shell() {
    if [ -n "$SHELL" ]; then
        SHELL_NAME=$(basename "$SHELL")
    else
        SHELL_NAME="bash"  # default fallback
    fi
    log_info "Detected shell: $SHELL_NAME"
}

# Detect Docker Compose (prefer v2 "docker compose", fallback to v1 "docker-compose")
detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE="docker compose"
        log_info "Using Docker Compose v2 (detected 'docker compose')."
        return 0
    fi
    if command_exists docker-compose; then
        COMPOSE="docker-compose"
        log_warning "Using Docker Compose v1 (detected 'docker-compose'). Consider upgrading to v2."
        return 0
    fi
    log_error "Docker Compose not found. Install Docker Compose v2 (preferred) or v1."
    log_error "Docs: https://docs.docker.com/compose/install/"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command_exists conda; then
        log_error "Conda is not installed. Please install Anaconda or Miniconda first:"
        log_error "https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    if ! command_exists docker; then
        log_error "Docker is not installed. Please install Docker first:"
        log_error "https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    detect_compose
    detect_shell
    
    log_success "All prerequisites are installed!"
}

# Create conda environment
setup_conda_environment() {
    log_info "Setting up conda environment..."
    
    if conda env list | grep -q "quants-lab"; then
        log_warning "Environment 'quants-lab' already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing environment..."
            conda env remove -n quants-lab -y
        else
            log_info "Using existing environment..."
            return 0
        fi
    fi
    
    log_info "Creating conda environment from environment.yml..."
    if ! conda env create -f environment.yml; then
        log_error "Failed to create conda environment."
        log_info "Try running: conda env remove -n quants-lab -y"
        log_info "Then run the installation again."
        exit 1
    fi
    log_success "Conda environment 'quants-lab' created successfully!"
}

# Install package in development mode
install_package() {
    log_info "Installing QuantsLab package in development mode..."
    
    eval "$(conda shell.$SHELL_NAME hook)"
    conda activate quants-lab
    
    pip install -e .
    
    log_success "QuantsLab package installed in development mode!"
}

# Setup databases
setup_databases() {
    log_info "Setting up databases..."
    
    read -p "Do you want to start the databases (MongoDB only) now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        log_info "Starting databases with Docker Compose..."
        
        if $COMPOSE -f docker-compose-db.yml ps | grep -q "Up"; then
            log_warning "Some database containers are already running."
            read -p "Do you want to restart them? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                $COMPOSE -f docker-compose-db.yml down
                $COMPOSE -f docker-compose-db.yml up -d
                DATABASES_STARTED=true
            else
                DATABASES_STARTED=true
            fi
        else
            $COMPOSE -f docker-compose-db.yml up -d
            DATABASES_STARTED=true
        fi
        
        log_success "Databases started successfully!"
        log_info "MongoDB connection: mongodb://admin:admin@localhost:27017/quants_lab"
    else
        log_warning "Databases not started. You can start them later with: make run-db"
        DATABASES_STARTED=false
    fi
}

# Test installation
test_installation() {
    log_info "Testing installation..."
    
    eval "$(conda shell.$SHELL_NAME hook)"
    conda activate quants-lab
    
    log_info "Testing CLI functionality..."
    if python cli.py --help > /dev/null 2>&1; then
        log_success "CLI is working!"
    else
        log_error "CLI test failed!"
        return 1
    fi
    
    log_info "Testing Python imports..."
    if python -c "from core.tasks.runner import TaskRunner; from app.tasks.data_collection.pools_screener import PoolsScreenerTask; print('Imports working!')" > /dev/null 2>&1; then
        log_success "Python imports are working!"
    else
        log_error "Import test failed!"
        return 1
    fi
    
    log_info "Testing task listing..."
    if python cli.py list-tasks > /dev/null 2>&1; then
        log_success "Task listing is working!"
    else
        log_error "Task listing test failed!"
        return 1
    fi
    
    log_success "All tests passed!"
}

# Create .env file if it doesn't exist
create_env_file() {
    if [ ! -f .env ]; then
        log_info "Creating .env file with placeholders..."
        cat > .env << 'EOF'
# Database Configuration
# MongoDB connection string (required)
MONGO_URI=mongodb://admin:admin@localhost:27017/quants_lab?authSource=admin&retryWrites=true&w=majority
MONGO_DATABASE=quants_lab

# Environment
ENVIRONMENT=development

# Notifiers Configuration (Optional - uncomment and fill to enable)

# Telegram
# TELEGRAM_ENABLED=true
# TELEGRAM_BOT_TOKEN=<your_telegram_bot_token>
# TELEGRAM_CHAT_ID=<your_telegram_chat_id>
# TELEGRAM_PARSE_MODE=HTML
# TELEGRAM_DISABLE_NOTIFICATION=false

# Email
# EMAIL_ENABLED=true
# EMAIL_SMTP_SERVER=smtp.gmail.com
# EMAIL_SMTP_PORT=587
# EMAIL_USERNAME=<your_email@example.com>
# EMAIL_PASSWORD=<your_app_specific_password>
# EMAIL_FROM=<sender_email@example.com>
# EMAIL_TO=<recipient1@example.com>,<recipient2@example.com>

# Hummingbot API
# HUMMINGBOT_API_HOST=<your_hummingbot_api_host_ip>

# Discord
# DISCORD_ENABLED=true
# DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/<your_webhook_id>/<your_webhook_token>

# Slack
# SLACK_ENABLED=true
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/<your_webhook_path>
# SLACK_CHANNEL=#general
EOF
        log_success ".env file created with placeholders!"
        log_warning "Please update the .env file with your actual configuration values"
        log_info "  - Update database credentials if using custom MongoDB setup"
        log_info "  - Configure notification services (Telegram, Email, Discord, Slack) as needed"
        log_info "  - Set Hummingbot API host if using remote Hummingbot instances"
    else
        log_info ".env file already exists, skipping..."
    fi
}

# Main installation flow
main() {
    echo
    log_info "🚀 Welcome to QuantsLab Installation!"
    echo
    log_info "This script will:"
    log_info "  1. Check prerequisites (conda, docker, docker compose)"
    log_info "  2. Create conda environment from environment.yml"
    log_info "  3. Install QuantsLab package in development mode"
    log_info "  4. Setup databases (optional)"
    log_info "  5. Create .env file with defaults"
    log_info "  6. Test the installation"
    echo
    
    read -p "Continue with installation? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Installation cancelled."
        exit 0
    fi
    
    check_prerequisites
    setup_conda_environment
    install_package
    setup_databases
    create_env_file
    test_installation
    
    echo
    log_success "🎉 QuantsLab installation completed successfully!"
    echo
    log_info "Next steps:"
    log_info "  1. Activate the environment: conda activate quants-lab"
    log_info "  2. Test the CLI: python cli.py --help"
    log_info "  3. List available tasks: python cli.py list-tasks"
    log_info "  4. Start Jupyter: jupyter lab"
    log_info "  5. Check the README.md for detailed usage instructions"
    echo
    log_info "Database access:"
    log_info "  MongoDB UI: http://localhost:28081/ (admin/changeme)"
    log_info "  Config file: config/database.yml"
    echo
    log_success "Happy coding! 🚀"
}

main "$@"
