# Makefile for backend Docker operations
# All commands must be run from the backend/ directory

# Configuration
IMAGE_NAME := mess-ai-backend
TAG := latest
PORT := 8000

.PHONY: help build run stop clean test logs

help:  ## Show this help message
	@echo "Backend Docker Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Build context: Parent directory (includes mess library)"
	@echo "Dockerfile: backend/Dockerfile"

build:  ## Build Docker image from parent directory
	@echo "Building $(IMAGE_NAME):$(TAG)..."
	cd .. && docker build -f backend/Dockerfile -t $(IMAGE_NAME):$(TAG) .
	@echo "✅ Build complete!"

run:  ## Run container locally
	@echo "Starting $(IMAGE_NAME) on port $(PORT)..."
	docker run -d \
		--name mess-backend \
		-p $(PORT):8000 \
		-e AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY} \
		-e AWS_REGION=$${AWS_REGION:-us-east-1} \
		$(IMAGE_NAME):$(TAG)
	@echo "✅ Container running at http://localhost:$(PORT)"
	@echo "   Docs: http://localhost:$(PORT)/docs"

run-dev:  ## Run container with auto-reload (mount app/ directory)
	@echo "Starting $(IMAGE_NAME) in dev mode..."
	docker run -it --rm \
		--name mess-backend-dev \
		-p $(PORT):8000 \
		-v $$(pwd)/app:/app/app \
		-e AWS_ACCESS_KEY_ID=$${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=$${AWS_SECRET_ACCESS_KEY} \
		-e AWS_REGION=$${AWS_REGION:-us-east-1} \
		$(IMAGE_NAME):$(TAG) \
		uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

stop:  ## Stop running container
	@echo "Stopping mess-backend..."
	docker stop mess-backend || true
	docker rm mess-backend || true
	@echo "✅ Container stopped"

logs:  ## Show container logs
	docker logs -f mess-backend

shell:  ## Open shell in running container
	docker exec -it mess-backend /bin/bash

test:  ## Test the running container
	@echo "Testing health endpoint..."
	curl -f http://localhost:$(PORT)/health && echo "✅ Health check passed" || echo "❌ Health check failed"

clean:  ## Remove container and image
	@echo "Cleaning up..."
	docker stop mess-backend 2>/dev/null || true
	docker rm mess-backend 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(TAG) 2>/dev/null || true
	@echo "✅ Cleanup complete"

push:  ## Push to registry (set REGISTRY env var)
	@if [ -z "$(REGISTRY)" ]; then \
		echo "❌ REGISTRY environment variable not set"; \
		echo "   Usage: REGISTRY=your-registry.com make push"; \
		exit 1; \
	fi
	docker tag $(IMAGE_NAME):$(TAG) $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME):$(TAG)
	@echo "✅ Pushed to $(REGISTRY)/$(IMAGE_NAME):$(TAG)"
