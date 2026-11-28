# Bucket arguments accept a name or its root s3:// URI. Return a bare name for
# AWS SDK calls, IAM resources and worker configuration.
anno_bucket_name() {
    local name="${1#s3://}" setting="$2"
    name="${name%/}"
    if [[ ! "$name" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ || "$name" == *..* ]]; then
        printf 'Set %s to a bucket name or s3://bucket/ without an object path.\n' "$setting" >&2
        return 2
    fi
    printf '%s\n' "$name"
}

anno_spot_config() {
    # Read the generated assignments as data. A setup for another bucket
    # must not supply its queue URL or worker configuration to this fleet.
    local chosen="${ANNO_SPOT_BUCKET:-${ANNO_S3_BUCKET:-}}" saved="" version="" saved_queue="" key value
    if [[ -f "$1" ]]; then
        while IFS='=' read -r key value; do
            case "$key" in
                ANNO_SPOT_CONFIG_VERSION) version="$value" ;;
                ANNO_SPOT_BUCKET) saved="$value" ;;
                ANNO_SPOT_QUEUE) saved_queue="$value" ;;
            esac
        done < "$1"
    fi
    local chosen_name="${chosen#s3://}" saved_name="${saved#s3://}"
    if [[ "$version" == 2 && -n "$saved_name" && ( -z "$chosen" || "${chosen_name%/}" == "${saved_name%/}" ) ]]; then
        chosen="${chosen:-$saved}"
        while IFS='=' read -r key value; do
            if [[ "$key" == ANNO_SPOT_QUEUE_URL && -n "${ANNO_SPOT_QUEUE:-}" && "$ANNO_SPOT_QUEUE" != "$saved_queue" ]]; then continue; fi
            case "$key" in
                ANNO_SPOT_REGION|ANNO_SPOT_QUEUE|ANNO_SPOT_QUEUE_URL|ANNO_SPOT_ROLE|ANNO_SPOT_PROFILE|ANNO_SPOT_TEMPLATE|ANNO_SPOT_CACHE_VOLUME|ANNO_SPOT_ACCOUNT)
                    if [[ -z "${!key:-}" ]]; then printf -v "$key" '%s' "$value"; fi ;;
            esac
        done < "$1"
    fi
    ANNO_SPOT_BUCKET="$chosen"
}
