.PHONY: ecr ecr-training ecr-inference \
        login \
        build build-training build-inference \
        build-arm64 build-training-arm64 build-inference-arm64 \
        push push-training push-inference \
        deploy-training deploy-inference help

include .env


TRAINING_REPO_NAME      := titanic-training
INFERENCE_REPO_NAME     := titanic-inference
TRAINING_IMAGE_TAG          := latest

ECR_BASE           := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
TRAINING_ECR_URI   := $(ECR_BASE)/$(TRAINING_REPO_NAME):$(IMAGE_TAG)
INFERENCE_ECR_URI  := $(ECR_BASE)/$(INFERENCE_REPO_NAME):$(IMAGE_TAG)

TRAINING_IMAGE  := $(TRAINING_REPO)-amd
INFERENCE_IMAGE := $(INFERENCE_REPO)-amd
TRAINING_IMAGE_ARM64  := $(TRAINING_REPO)-arm
INFERENCE_IMAGE_ARM64 := $(INFERENCE_REPO)-arm

MAKEFLAGS += --silent

help: ## Show the list of available commands
	@echo "Available commands:"
	@grep -h -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  🔹 \033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ── ECR Management ────────────────────────────────────────────────────────────

ecr-training: ## Create the ECR repository for training
	aws ecr create-repository \
		--repository-name $(TRAINING_REPO) \
		--region $(AWS_REGION) \
		--image-scanning-configuration scanOnPush=true \
		|| true

ecr-inference: ## Create the ECR repository for inference
	aws ecr create-repository \
		--repository-name $(INFERENCE_REPO) \
		--region $(AWS_REGION) \
		--image-scanning-configuration scanOnPush=true \
		|| true

ecr: ecr-training ecr-inference ## Create all ECR repositories

# ── Build AMD64 (Production Standard) ─────────────────────────────────────────

build-training: ## Build the training image (linux/amd64)
	docker buildx build --platform linux/amd64 \
		-f Dockerfile.training \
		-t $(LOCAL_TRAINING_AMD64):$(IMAGE_TAG) --load .

build-inference: ## Build the inference image (linux/amd64)
	docker buildx build --platform linux/amd64 \
		-f Dockerfile.inference \
		-t $(LOCAL_INFERENCE_AMD64):$(IMAGE_TAG) --load .

build: build-training build-inference ## Build all AMD64 images

# ── Build ARM64 (Local Testing) ───────────────────────────────────────────────

build-training-arm64: ## Build the training image (linux/arm64)
	docker buildx build --platform linux/arm64 \
		-f Dockerfile.training \
		-t $(LOCAL_TRAINING_ARM64):$(IMAGE_TAG) --load .

build-inference-arm64: ## Build the inference image (linux/arm64)
	docker buildx build --platform linux/arm64 \
		-f Dockerfile.inference \
		-t $(LOCAL_INFERENCE_ARM64):$(IMAGE_TAG) --load .

build-arm64: build-training-arm64 build-inference-arm64 ## Build all ARM64 images

# ── Push to ECR (Locked to AMD64) ──────────────────────────────────────────────

login: ## Authenticate Docker to AWS ECR
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin $(ECR_BASE)

push-training: login build-training ## Force AMD64 build, tag and push training image
	docker tag $(LOCAL_TRAINING_AMD64):$(IMAGE_TAG) $(TRAINING_ECR_URI)
	docker push $(TRAINING_ECR_URI)

push-inference: login build-inference ## Force AMD64 build, tag and push inference image
	docker tag $(LOCAL_INFERENCE_AMD64):$(IMAGE_TAG) $(INFERENCE_ECR_URI)
	docker push $(INFERENCE_ECR_URI)

push: push-training push-inference ## Push all images to ECR

# ── Deployment ────────────────────────────────────────────────────────────────

deploy-training: ecr-training push-training ## Full deploy: ECR setup + AMD64 Build + Push

deploy-inference: ecr-inference push-inference ## Full deploy: ECR setup + AMD64 Build + Push

deploy: deploy-training deploy-inference ## Full deploy for both images

# ── Run Local Containers ──────────────────────────────────────────────────────

run-training: build-training ## Run training container locally (AMD64)
	docker run -it --rm \
		--name training-container \
		$(LOCAL_TRAINING_AMD64):$(IMAGE_TAG)

run-inference: build-inference ## Run inference container locally (AMD64)
	docker run -it --rm \
		--name inference-container \
		-p 8080:8080 \
		$(LOCAL_INFERENCE_AMD64):$(IMAGE_TAG)

run-training-arm64: build-training-arm64 ## Run training container locally (ARM64)
	docker run -it --rm \
		--name training-container \
		$(LOCAL_TRAINING_ARM64):$(IMAGE_TAG)

run-inference-arm64: build-inference-arm64 ## Run inference container locally (ARM64)
	docker run -it --rm \
		--name inference-container \
		-p 8080:8080 \
		$(LOCAL_INFERENCE_ARM64):$(IMAGE_TAG)