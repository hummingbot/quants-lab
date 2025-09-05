#!/bin/bash

# QuantsLab Uninstallation Script
# This script completely removes the QuantsLab development environment

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

# Check if we're in the right directory
check_project_directory() {
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "core" ]] || [[ ! -d "app" ]]; then
        log_error "This doesn't appear to be the QuantsLab project directory."
        log_error "Please run this script from the QuantsLab root directory."
        exit 1
    fi
}

# Stop running services
stop_services() {
    log_info "Stopping running services..."
    
    # Stop task runner if running
    if command_exists docker && docker ps --format "table {{.Names}}" | grep -q "quants-lab-task-runner"; then
        log_info "Stopping task runner..."
        docker stop quants-lab-task-runner 2>/dev/null || true
        docker rm quants-lab-task-runner 2>/dev/null || true
    fi
    
    # Stop databases if running
    if [[ -f "docker-compose-db.yml" ]]; then
        log_info "Stopping databases..."
        docker-compose -f docker-compose-db.yml down 2>/dev/null || true
        log_success "Database containers stopped"
    fi
    
    # Remove any other containers with quants-lab prefix
    if command_exists docker; then
        CONTAINERS=$(docker ps -a --format "{{.Names}}" | grep "quants-lab" || true)
        if [[ -n "$CONTAINERS" ]]; then
            log_info "Removing remaining QuantsLab containers..."
            echo "$CONTAINERS" | xargs docker rm -f 2>/dev/null || true
        fi
    fi
    
    log_success "All services stopped"
}

# Remove conda environment
remove_conda_environment() {
    log_info "Removing conda environment..."
    
    if ! command_exists conda; then
        log_warning "Conda not found, skipping environment removal"
        return 0
    fi
    
    # Check if environment exists
    if conda env list | grep -q "quants-lab"; then
        log_info "Found 'quants-lab' conda environment"
        
        # Deactivate if currently active
        if [[ "$CONDA_DEFAULT_ENV" == "quants-lab" ]]; then
            log_info "Deactivating quants-lab environment..."
            eval "$(conda shell.bash hook)"
            conda deactivate || true
        fi
        
        # Remove environment
        log_info "Removing conda environment 'quants-lab'..."
        conda env remove -n quants-lab -y
        log_success "Conda environment removed"
    else
        log_info "No 'quants-lab' conda environment found"
    fi
}

# Remove Docker images
remove_docker_images() {
    log_info "Removing Docker images and volumes..."
    
    if ! command_exists docker; then
        log_warning "Docker not found, skipping image cleanup"
        return 0
    fi
    
    # Remove custom images
    IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "quants-lab" || true)
    if [[ -n "$IMAGES" ]]; then
        log_info "Removing QuantsLab Docker images..."
        echo "$IMAGES" | xargs docker rmi -f 2>/dev/null || true
        log_success "Docker images removed"
    fi
    
    # Remove volumes
    VOLUMES=$(docker volume ls --format "{{.Name}}" | grep "quants-lab" || true)
    if [[ -n "$VOLUMES" ]]; then
        log_info "Removing QuantsLab Docker volumes..."
        echo "$VOLUMES" | xargs docker volume rm -f 2>/dev/null || true
        log_success "Docker volumes removed"
    fi
    
    # Remove networks
    NETWORKS=$(docker network ls --format "{{.Name}}" | grep "quants-lab" || true)
    if [[ -n "$NETWORKS" ]]; then
        log_info "Removing QuantsLab Docker networks..."
        echo "$NETWORKS" | xargs docker network rm 2>/dev/null || true
        log_success "Docker networks removed"
    fi
}

# Clean generated files and directories
clean_generated_files() {
    log_info "Cleaning generated files and directories..."
    
    # Remove Python cache files
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove build artifacts
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    
    # Remove output directories (ask user first)
    if [[ -d "outputs" ]]; then
        read -p "Remove outputs directory? This will delete all generated reports and data. (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf outputs/
            log_success "Outputs directory removed"
        else
            log_info "Keeping outputs directory"
        fi
    fi
    
    # Remove app/data if it exists (ask user first)
    if [[ -d "app/data" ]]; then
        read -p "Remove app/data directory? This will delete cached application data. (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf app/data/
            log_success "App data directory removed"
        else
            log_info "Keeping app/data directory"
        fi
    fi
    
    # Remove .env file (ask user first)
    if [[ -f ".env" ]]; then
        read -p "Remove .env file? This will delete your environment configuration. (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f .env
            log_success ".env file removed"
        else
            log_info "Keeping .env file"
        fi
    fi
    
    # Remove Jupyter checkpoints
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove log files
    rm -f *.log 2>/dev/null || true
    
    log_success "Generated files cleaned"
}

# Remove pip package installation
remove_pip_package() {
    log_info "Removing pip package installation..."
    
    # Try to uninstall the package from current environment
    if pip list | grep -q "quants-lab"; then
        log_info "Uninstalling quants-lab package..."
        pip uninstall quants-lab -y 2>/dev/null || true
        log_success "Package uninstalled"
    else
        log_info "Package not found in current environment"
    fi
}

# Show final cleanup summary
show_cleanup_summary() {
    log_info "Cleanup Summary:"
    echo "  ✅ Services stopped"
    echo "  ✅ Conda environment removed"
    echo "  ✅ Docker resources cleaned"
    echo "  ✅ Generated files cleaned"
    echo "  ✅ Package uninstalled"
    echo
    log_info "Files preserved (if they exist):"
    echo "  📁 Source code (core/, app/, config/)"
    echo "  📁 Configuration files (config/)"
    echo "  📁 Documentation files (*.md)"
    echo "  📁 Research notebooks (app/research_notebooks/)"
    
    if [[ -d "outputs" ]]; then
        echo "  📁 outputs/ (user choice)"
    fi
    if [[ -d "app/data" ]]; then
        echo "  📁 app/data/ (user choice)"
    fi
    if [[ -f ".env" ]]; then
        echo "  📄 .env (user choice)"
    fi
}

# Main uninstallation flow
main() {
    echo
    log_info "🧹 QuantsLab Uninstallation Script"
    echo
    log_warning "This script will remove:"
    log_warning "  - Conda environment 'quants-lab'"
    log_warning "  - Docker containers, images, and volumes"
    log_warning "  - Python cache files and build artifacts"
    log_warning "  - Running services and processes"
    echo
    log_info "Your source code and configuration files will be preserved."
    echo
    
    read -p "Are you sure you want to uninstall QuantsLab? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Uninstallation cancelled."
        exit 0
    fi
    
    echo
    log_info "Starting uninstallation process..."
    
    # Run uninstallation steps
    check_project_directory
    stop_services
    remove_conda_environment
    remove_pip_package
    remove_docker_images
    clean_generated_files
    
    echo
    log_success "🎉 QuantsLab uninstallation completed!"
    echo
    show_cleanup_summary
    echo
    log_info "To reinstall QuantsLab, simply run: ./install.sh"
    echo
}

# Handle script interruption
cleanup() {
    echo
    log_warning "Uninstallation interrupted!"
    log_info "You may need to manually clean up any partially removed components."
    exit 1
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"