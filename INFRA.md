# Unity System Infrastructure Documentation

## 🏗️ Overview

The Unity system is a comprehensive multi-channel communication platform that dynamically provisions infrastructure for AI assistants. Each assistant gets its own dedicated communication channels, cloud resources, and notification systems.

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [User Flow](#-user-flow)
- [Infrastructure Components](#️-infrastructure-components)
- [GKE Architecture](#-gke-architecture)
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
POST /assistant → Orchestra → Communications Endpoints → Infrastructure Provisioning → GKE Container Activation
```

1. **Assistant Creation**: User calls the `/assistant` POST endpoint in Orchestra
2. **Database Entry**: Assistant is created in the database
3. **Infrastructure Provisioning**: Orchestra calls Communications repo endpoints to provision resources
4. **GKE Container Activation**: Startup topic receives user-specific details and activates an idle container
5. **Idle Container Replacement**: A new idle container is created to maintain availability

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
| **Pub/Sub Startup Topic** | `/infra/pubsub/startup` | `unity-startup` | Container activation |
| **GKE Job** | `/infra/gke/job` | `unity-{assistant_id}` | Assistant runtime |

### Cleanup

All infrastructure can be removed via the `DELETE /assistant` endpoint in Orchestra, which reverses all provisioning steps.

## 🐳 GKE Architecture

### Idle Container System

The Unity system now uses Google Kubernetes Engine (GKE) instead of individual Cloud Run instances:

- **Always-On Idle Container**: GKE maintains one idle container at all times
- **No User Context**: Idle containers start without user-specific environment variables
- **Startup Topic Listener**: Idle containers listen to the `unity-startup` Pub/Sub topic
- **Dynamic Activation**: When a request comes in, the startup topic receives user-specific details
- **Container Activation**: The idle container receives the user context and becomes active
- **Replacement**: A new idle container is immediately created to maintain availability

### Container Lifecycle

```
Idle Container → Startup Topic → User Context → Active Container → New Idle Container
```

1. **Idle State**: Container runs without user-specific configuration
2. **Activation**: Startup topic delivers user-specific environment variables
3. **Active State**: Container processes requests for the specific user
4. **Replacement**: New idle container is spawned for future requests

## 🔗 Webhook System

### Cloud Functions (Communications Repo)

Located in the `adapters` folder, these functions serve as webhooks for external services:

| Function | Status | Purpose |
|----------|--------|---------|
| `email-notification-processor` | ⏸️ Disabled | Process incoming email notifications |
| `twilio-call-webhook` | ✅ Active | Handle incoming voice calls |
| `twilio-msg-webhook` | ✅ Active | Process incoming SMS messages |
| `twilio-whatsapp-webhook` | ✅ Active | Handle incoming WhatsApp messages |
| `idle-job-renewer` | ✅ Active | Renew idle GKE job daily |

### Notification Flow

```
External Service → Cloud Function → Pub/Sub Topic → GKE Container → Outbound Action
```

1. **Incoming Notification**: External service (Twilio, etc.) sends webhook to cloud function
2. **Service Activation**: Cloud function calls `/start` endpoint on the wrapper app in `/`
3. **Message Routing**: Notification is sent to the appropriate Pub/Sub topic (`unity-{assistant_id}`)
4. **Container Processing**: GKE container processes the message
5. **Outbound Action**: Service uses Communications endpoints for responses

## 🚀 Deployment & CI/CD

### Unity Repository

| File | Purpose |
|------|---------|
| `cloudbuild.yaml` | Production environment deployment |
| `cloudbuild-staging.yaml` | Staging environment deployment |

### Communications Repository

| File/Directory | Purpose |
|----------------|---------|
| `cloudbuild/*.yaml` | Build configuration for all cloud functions |
| `.github/deploy.yaml` | Build and deploy the communications Cloud Run service |

### Environment Workflows

- **Staging**: Uses `cloudbuild-staging.yaml` for testing and validation
- **Production**: Uses `cloudbuild.yaml` for live deployment
- **Daily Maintenance**: Infra adapter automatically renews idle GKE jobs

## 📁 Repository Structure

```
unity/
├── cloudbuild.yaml               # Docker image build configuration (staging)
├── cloudbuild-staging.yaml       # Staging deployment configuration

communications/
├── adapters/                     # Cloud function webhooks
│   ├── email-notification-processor
│   ├── twilio-call-webhook
│   ├── twilio-msg-webhook
│   ├── twilio-whatsapp-webhook
│   └── idle-job-renewer           # Daily GKE job renewal
├── cloudbuild/                   # Cloud function build configs
└── .github/deploy.yaml           # CI/CD for communications service

orchestra/
└── /assistant                    # Main orchestration endpoints
```

## 🔧 Technical Details


### Communication Flow

1. **Inbound**: External Service → Cloud Function → Pub/Sub → GKE Container
2. **Outbound**: GKE Container → Communications API → External Service

### Scalability

- GKE provides better resource utilization than individual Cloud Run instances
- Idle containers ensure immediate response to new requests
- Automatic container replacement maintains high availability
- Daily job renewal prevents resource exhaustion

### Startup Topic Details

- **Topic Name**: `unity-startup`
- **Payload**: User-specific environment variables and configuration
- **Listeners**: All idle containers in the GKE cluster
- **Activation**: Container receives payload and switches to active mode

---
