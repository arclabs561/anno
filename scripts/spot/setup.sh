#!/bin/bash
# One-time setup for AWS spot evaluation infrastructure
#
# Uses an operator-provisioned S3 bucket. Creates the IAM role, SQS queue, EBS volume and launch template.
#
# Usage:
#   ./scripts/spot/setup.sh
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - jq installed

set -euo pipefail

REGION="${ANNO_SPOT_REGION:-us-east-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../s3_bucket.sh"

EBS_CACHE_SIZE_GB="${ANNO_SPOT_CACHE_SIZE:-50}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prereqs() {
    log_info "Checking prerequisites..."
    
    if ! command -v aws &>/dev/null; then
        log_error "AWS CLI not found. Install: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    if ! command -v jq &>/dev/null; then
        log_error "jq not found. Install: brew install jq"
        exit 1
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure'"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_info "AWS Account: $ACCOUNT_ID"
    log_info "Region: $REGION"
}

create_iam_role() {
    log_info "Creating IAM role: $ROLE_NAME"
    
    # Trust policy for EC2
    cat > "$SETUP_DIR/trust-policy.json" << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    # Check if role exists
    if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
        log_warn "Role $ROLE_NAME already exists, skipping creation"
    else
        aws iam create-role \
            --role-name "$ROLE_NAME" \
            --assume-role-policy-document "file://$SETUP_DIR/trust-policy.json" \
            --description "Role for anno evaluation spot instances"
        log_info "Created role: $ROLE_NAME"
    fi
    
    # Policy for S3, SQS, EC2 (for EBS attach), CloudWatch
    cat > "$SETUP_DIR/role-policy.json" << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::${BUCKET}",
            "Condition": {"StringLike": {"s3:prefix": ["datasets/*", "models/*"]}}
        },
        {
            "Effect": "Allow",
            "Action": "s3:GetObject",
            "Resource": ["arn:aws:s3:::${BUCKET}/datasets/*", "arn:aws:s3:::${BUCKET}/models/*", "arn:aws:s3:::${BUCKET}/src/anno-src.tar.gz"]
        },
        {
            "Effect": "Allow",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::${BUCKET}/results/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sqs:ReceiveMessage",
                "sqs:DeleteMessage",
                "sqs:GetQueueAttributes",
                "sqs:ChangeMessageVisibility"
            ],
            "Resource": "arn:aws:sqs:${REGION}:${ACCOUNT_ID}:${QUEUE_NAME}"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:AttachVolume",
                "ec2:DetachVolume",
                "ec2:DescribeVolumes"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/anno-eval/*"
        }
    ]
}
EOF

    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "anno-eval-policy" \
        --policy-document "file://$SETUP_DIR/role-policy.json"
    log_info "Attached inline policy to role"
    
    # Attach SSM managed policy for SSM Agent connectivity
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
    log_info "Attached SSM managed policy to role"
    
    # Create instance profile
    if ! aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" &>/dev/null; then
        aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE"
        aws iam add-role-to-instance-profile \
            --instance-profile-name "$INSTANCE_PROFILE" \
            --role-name "$ROLE_NAME"
        log_info "Created instance profile: $INSTANCE_PROFILE"
    else
        log_warn "Instance profile $INSTANCE_PROFILE already exists"
    fi
}

create_sqs_queue() {
    log_info "Creating SQS queue: $QUEUE_NAME"
    
    # Check if queue exists
    if aws sqs get-queue-url --queue-name "$QUEUE_NAME" --region "$REGION" &>/dev/null; then
        log_warn "Queue $QUEUE_NAME already exists"
        QUEUE_URL=$(aws sqs get-queue-url --queue-name "$QUEUE_NAME" --region "$REGION" --query QueueUrl --output text)
    else
        QUEUE_URL=$(aws sqs create-queue \
            --queue-name "$QUEUE_NAME" \
            --region "$REGION" \
            --attributes '{
                "VisibilityTimeout": "600",
                "MessageRetentionPeriod": "86400",
                "ReceiveMessageWaitTimeSeconds": "20"
            }' \
            --query QueueUrl --output text)
        log_info "Created queue: $QUEUE_URL"
    fi
}

create_ebs_volume() {
    log_info "Creating EBS cache volume (${EBS_CACHE_SIZE_GB}GB gp3)..."
    
    # Get default AZ
    DEFAULT_AZ=$(aws ec2 describe-availability-zones \
        --region "$REGION" \
        --query 'AvailabilityZones[0].ZoneName' \
        --output text)
    
    # Check for existing tagged volume
    EXISTING_VOL=$(aws ec2 describe-volumes \
        --region "$REGION" \
        --filters "Name=tag:Name,Values=anno-eval-cache" "Name=status,Values=available" \
        --query 'Volumes[0].VolumeId' --output text 2>/dev/null || echo "None")
    
    if [[ "$EXISTING_VOL" != "None" && "$EXISTING_VOL" != "null" ]]; then
        log_warn "Cache volume already exists: $EXISTING_VOL"
        VOLUME_ID="$EXISTING_VOL"
    else
        VOLUME_ID=$(aws ec2 create-volume \
            --region "$REGION" \
            --availability-zone "$DEFAULT_AZ" \
            --volume-type gp3 \
            --size "$EBS_CACHE_SIZE_GB" \
            --iops 3000 \
            --throughput 125 \
            --tag-specifications "ResourceType=volume,Tags=[{Key=Name,Value=anno-eval-cache},{Key=Project,Value=anno},{Key=Component,Value=eval-cache},{Key=Environment,Value=eval},{Key=ManagedBy,Value=anno-spot-scripts},{Key=CostCenter,Value=ml-eval}]" \
            --query VolumeId --output text)
        log_info "Created volume: $VOLUME_ID"
    fi
}

worker_user_data() {
    printf '#!/bin/bash\nexport ANNO_SPOT_BUCKET=%q\nexport ANNO_SPOT_QUEUE=%q\nexport ANNO_SPOT_QUEUE_URL=%q\nexport ANNO_SPOT_REGION=%q\n' "$BUCKET" "$QUEUE_NAME" "$QUEUE_URL" "$REGION"
    cat <<'USERDATA'
set -ex

# Set HOME explicitly for root context
export HOME=/root
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"

exec > >(tee /var/log/anno-worker-init.log) 2>&1
echo "=== Anno Worker Init $(date) ==="
echo "HOME=$HOME"

# Install dependencies (AL2023)
# Note: gcc-c++ is required for ONNX runtime compilation
if command -v dnf &>/dev/null; then
    dnf install -y git gcc gcc-c++ openssl-devel pkg-config jq cmake
fi

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$CARGO_HOME/env"
export PATH="$CARGO_HOME/bin:$PATH"
cargo --version

REGION="$ANNO_SPOT_REGION"

# Download source from S3 (repo is private, can't git clone)
cd $HOME
aws s3 cp "s3://$ANNO_SPOT_BUCKET/src/anno-src.tar.gz" . --region $REGION
mkdir -p anno
tar xzf anno-src.tar.gz -C anno
cd anno

# Set up persistent cache directories (shared with worker.sh)
export ANNO_CACHE_MOUNT=/opt/cache
export CARGO_TARGET_DIR="$ANNO_CACHE_MOUNT/target"
export SCCACHE_DIR="$ANNO_CACHE_MOUNT/sccache"
export ANNO_CACHE_DIR="$ANNO_CACHE_MOUNT"
export ANNO_SOURCE_DIR="$HOME/anno"
mkdir -p "$CARGO_TARGET_DIR" "$SCCACHE_DIR"

# Build anno with ONNX support (can be slow on fresh instances)
# ONNX enables ML backends: gliner, gliner_multitask, nuner, w2ner
echo "Building anno with ONNX support..."
cargo build --locked --release -p anno-cli --bin anno --features "eval onnx" 2>&1 | tail -20
cargo build --locked --release -p anno-eval --example public_benchmark_report --features "eval onnx" 2>&1 | tail -20

# Sync NER models from S3 (large; needed for ML backends)
echo "Syncing NER models from S3..."
mkdir -p "$ANNO_CACHE_MOUNT/models"
export HF_HOME="$ANNO_CACHE_MOUNT/models"
aws s3 sync "s3://$ANNO_SPOT_BUCKET/models/ner/" "$ANNO_CACHE_MOUNT/models/ner/" --region $REGION --quiet || true
echo "Models synced: $(du -sh $ANNO_CACHE_MOUNT/models 2>/dev/null | cut -f1)"

# Enable S3 fallback for any missing datasets
export ANNO_S3_CACHE=1
export ANNO_S3_BUCKET="$ANNO_SPOT_BUCKET"

# Signal ready
touch /tmp/anno-worker-ready
echo "=== Worker Ready $(date) ==="

# Start worker with env vars
export ANNO_SPOT_REGION="$REGION"

./scripts/spot/worker.sh >> /var/log/anno-worker.log 2>&1 &

echo "Worker started with PID $!"
USERDATA
}

create_launch_template() {
    log_info "Creating launch template: $LAUNCH_TEMPLATE_NAME"

    # Get latest Amazon Linux 2023 AMI
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)

    log_info "Using AMI: $AMI_ID (Amazon Linux 2023)"

    USER_DATA=$(worker_user_data | base64 | tr -d '\n')

    # Create or update launch template
    cat > "$SETUP_DIR/launch-template.json" << EOF
{
    "LaunchTemplateName": "$LAUNCH_TEMPLATE_NAME",
    "LaunchTemplateData": {
        "ImageId": "$AMI_ID",
        "InstanceType": "c7i.xlarge",
        "IamInstanceProfile": {
            "Name": "$INSTANCE_PROFILE"
        },
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/xvda",
                "Ebs": {
                    "VolumeSize": 100,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": true
                }
            }
        ],
        "UserData": "$USER_DATA",
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "anno-eval-worker"},
                    {"Key": "Project", "Value": "anno"},
                    {"Key": "Component", "Value": "eval-worker"},
                    {"Key": "Environment", "Value": "eval"},
                    {"Key": "ManagedBy", "Value": "anno-spot-scripts"},
                    {"Key": "CostCenter", "Value": "ml-eval"}
                ]
            },
            {
                "ResourceType": "volume",
                "Tags": [
                    {"Key": "Name", "Value": "anno-eval-worker-volume"},
                    {"Key": "Project", "Value": "anno"},
                    {"Key": "Component", "Value": "eval-worker"},
                    {"Key": "CostCenter", "Value": "ml-eval"}
                ]
            }
        ],
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate"
            }
        }
    }
}
EOF

    # Delete existing template if it exists
    aws ec2 delete-launch-template --launch-template-name "$LAUNCH_TEMPLATE_NAME" --region "$REGION" 2>/dev/null || true
    
    aws ec2 create-launch-template \
        --region "$REGION" \
        --cli-input-json "file://$SETUP_DIR/launch-template.json"
    
    log_info "Created launch template: $LAUNCH_TEMPLATE_NAME"
}

save_config() {
    log_info "Saving configuration..."
    
    
    cat > "$SCRIPT_DIR/config.env" << EOF
# Anno Spot Evaluation Configuration
# Generated: $(date -uIs)

ANNO_SPOT_CONFIG_VERSION=2
ANNO_SPOT_REGION=$REGION
ANNO_SPOT_BUCKET=$BUCKET
ANNO_SPOT_QUEUE=$QUEUE_NAME
ANNO_SPOT_QUEUE_URL=$QUEUE_URL
ANNO_SPOT_ROLE=$ROLE_NAME
ANNO_SPOT_PROFILE=$INSTANCE_PROFILE
ANNO_SPOT_TEMPLATE=$LAUNCH_TEMPLATE_NAME
ANNO_SPOT_CACHE_VOLUME=$VOLUME_ID
ANNO_SPOT_ACCOUNT=$ACCOUNT_ID
EOF

    log_info "Configuration saved to scripts/spot/config.env"
}

main() {
    if [[ ${1:-} == --help || ${1:-} == -h || ${1:-} == help ]]; then
        echo 'Usage: ANNO_SPOT_BUCKET=bucket scripts/spot/setup.sh'
        return
    fi
    BUCKET=$(anno_bucket_name "${ANNO_SPOT_BUCKET:-${ANNO_S3_BUCKET:-}}" "ANNO_SPOT_BUCKET or ANNO_S3_BUCKET")
    QUEUE_NAME="${ANNO_SPOT_QUEUE:-anno-eval-${BUCKET//./_}}"
    local binding
    binding=$(python3 -c 'import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:16])' "$BUCKET")
    ROLE_NAME="anno-eval-$binding-role"
    INSTANCE_PROFILE="anno-eval-$binding-profile"
    LAUNCH_TEMPLATE_NAME="anno-eval-worker-${BUCKET}"
    SETUP_DIR=$(mktemp -d)
    trap 'rm -rf -- "$SETUP_DIR"' EXIT
    echo "========================================"
    echo "  Anno Spot Evaluation Setup"
    echo "========================================"
    echo ""
    
    check_prereqs
    create_iam_role
    create_sqs_queue
    create_ebs_volume
    create_launch_template
    save_config
    
    echo ""
    echo "========================================"
    echo "  Setup Complete"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run: just spot-eval"
    echo "  2. Monitor: just spot-status"
    echo "  3. Results: just spot-results"
    echo ""
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi

