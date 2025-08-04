.PHONY: help install test run clean docker-build docker-run docker-compose-up docker-compose-down lint format type-check

# Default target
help:
	@echo "Dental X-Ray Detection API - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      - Install dependencies"
	@echo "  run          - Run the development server"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  type-check   - Run type checking"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-compose-up   - Start with Docker Compose"
	@echo "  docker-compose-down - Stop Docker Compose"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean up generated files"
	@echo "  help         - Show this help message"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the development server
run:
	python run.py --reload

# Run tests
test:
	pytest tests/ -v --cov=app --cov-report=html

# Run linting
lint:
	flake8 app/ tests/
	pylint app/

# Format code
format:
	black app/ tests/
	isort app/ tests/

# Type checking
type-check:
	mypy app/

# Clean up generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Docker commands
docker-build:
	docker build -t dental-detection-api .

docker-run:
	docker run -p 8000:8000 dental-detection-api

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Production commands
docker-compose-up-prod:
	docker-compose --profile production up -d

docker-compose-up-monitoring:
	docker-compose --profile monitoring up -d

# Development setup
dev-setup: install
	@echo "Development environment setup complete!"
	@echo "Run 'make run' to start the development server"

# Full test suite
test-full: lint type-check test
	@echo "All tests passed!"

# Production build
prod-build: clean
	docker build -t dental-detection-api:latest .
	@echo "Production build complete!"

# Quick start for development
quick-start: install run 