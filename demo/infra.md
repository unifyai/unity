# Unity System Infrastructure Documentation

## 🏗️ Overview

The Unity system is a comprehensive multi-channel communication platform that dynamically provisions infrastructure for AI assistants. Each assistant gets its own dedicated communication channels, cloud resources, and notification systems.

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [User Flow](#-user-flow)
- [Infrastructure Components](#️-infrastructure-components)
- [Webhook System](#-webhook-system)
- [Deployment & CI/CD](#-deployment--cicd)
- [Repository Structure](#-repository-structure)

## 🏛️ System Architecture

The system consists of three main repositories:

1. **Orchestra** (`@https://github.com/unifyai/orchestra`) - Main orchestration service
2. **Communications** (`@https://github.com/unifyai/communication`) - Communication infrastructure APIs
3. **Unity** (current repo) - Assistant runtime and deployment configurations

## 🔄 User Flow

The assistant creation process follows this sequence:

```
POST /assistant → Orchestra → Communications Endpoints → Infrastructure Provisioning
```

1. **Assistant Creation**: User calls the `/assistant` POST endpoint in Orchestra
2. **Database Entry**: Assistant is created in the database
3. **Infrastructure Provisioning**: Orchestra calls Communications repo endpoints to provision resources
4. **Service Deployment**: Cloud Run service is deployed and configured

## 🛠️ Infrastructure Components

Each assistant gets the following dedicated infrastructure:

### Communication Channels

| Component | Endpoint | Purpose |
|-----------|----------|---------|
| **Email Address** | `/email/create` | Dedicated email for the assistant |
| **Phone Number** | `/phone/create` | Voice and SMS communication |
| **WhatsApp Sender** | `/whatsapp/create` | WhatsApp messaging capability |

### Cloud Infrastructure

| Component | Endpoint | Naming Convention | Purpose |
|-----------|----------|-------------------|---------|
| **Pub/Sub Topic** | `/infra/pubsub/topic` | `unity-{assistant_id}` | Notification routing |
| **Cloud Run Service** | `/infra/job/create` | `unity-{assistant_id}` | Assistant runtime |

### Cleanup

All infrastructure can be removed via the `DELETE /assistant` endpoint in Orchestra, which reverses all provisioning steps.

## 🔗 Webhook System

### Cloud Functions (Communications Repo)

Located in the `adapters` folder, these functions serve as webhooks for external services:

| Function | Status | Purpose |
|----------|--------|---------|
| `email-notification-processor` | ⏸️ Disabled | Process incoming email notifications |
| `twilio-call-webhook` | ✅ Active | Handle incoming voice calls |
| `twilio-msg-webhook` | ✅ Active | Process incoming SMS messages |
| `twilio-whatsapp-webhook` | ✅ Active | Handle incoming WhatsApp messages |

### Notification Flow

```
External Service → Cloud Function → Pub/Sub Topic → Cloud Run Service → Outbound Action
```

1. **Incoming Notification**: External service (Twilio, etc.) sends webhook to cloud function
2. **Service Activation**: Cloud function calls `/start` endpoint on the wrapper app in `demo/`
3. **Message Routing**: Notification is sent to the appropriate Pub/Sub topic (`unity-{assistant_id}`)
4. **Processing**: Cloud Run service processes the message
5. **Outbound Action**: Service uses Communications endpoints for responses

## 🚀 Deployment & CI/CD

### Unity Repository

| File | Purpose |
|------|---------|
| `cloudbuild.yaml` | Build new Docker images with updated code |
| `cloudbuild-deploy.yaml` | Deploy updated images to all Cloud Run instances |

### Communications Repository

| File/Directory | Purpose |
|----------------|---------|
| `cloudbuild/*.yaml` | Build configuration for all cloud functions |
| `.github/deploy.yaml` | Build and deploy the communications Cloud Run service |

## 📁 Repository Structure

```
unity/
├── demo/                          # Assistant runtime and deployment configs
│   ├── INFRA.md                  # Original infrastructure notes
│   └── INFRASTRUCTURE.md         # This documentation
├── cloudbuild.yaml               # Docker image build configuration
└── cloudbuild-deploy.yaml        # Deployment configuration

communications/
├── adapters/                     # Cloud function webhooks
│   ├── email-notification-processor/
│   ├── twilio-call-webhook/
│   ├── twilio-msg-webhook/
│   └── twilio-whatsapp-webhook/
├── cloudbuild/                   # Cloud function build configs
└── .github/deploy.yaml           # CI/CD for communications service

orchestra/
└── /assistant                    # Main orchestration endpoints
```

## 🔧 Technical Details

### Resource Naming

All resources follow the naming convention: `unity-{assistant_id}`

### Communication Flow

1. **Inbound**: External Service → Cloud Function → Pub/Sub → Cloud Run
2. **Outbound**: Cloud Run → Communications API → External Service

### Scalability

- Each assistant is isolated with dedicated resources
- Cloud Run provides automatic scaling based on demand
- Pub/Sub ensures reliable message delivery

---
