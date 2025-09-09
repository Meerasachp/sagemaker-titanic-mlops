#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-east-1}"
ENDPOINT=""
MODEL_NAME=""
VARIANT_NAME="AllTraffic"
MEM_MB=2048
MAX_CONCURRENCY=5
TEST_ROW="3,1,34,0,0,7.8292,2"

usage() {
  cat <<USAGE
Usage: $0 --endpoint <name> --model <model-name> [--variant <variant-name>] [--region <aws-region>] [--mem <MB>] [--maxc <N>]
Example:
  $0 --endpoint titanic-xgboost-endpoint-1757260241 \
     --model sagemaker-xgboost-2025-09-07-15-50-46-469 \
     --region us-east-1 --mem 2048 --maxc 5
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --variant) VARIANT_NAME="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --mem) MEM_MB="$2"; shift 2 ;;
    --maxc) MAX_CONCURRENCY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$ENDPOINT" || -z "$MODEL_NAME" ]]; then
  echo "ERROR: --endpoint and --model are required."; usage; exit 1
fi

export AWS_DEFAULT_REGION="$REGION"
aws sts get-caller-identity >/dev/null

ECONFIG_OLD=$(aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" --region "$REGION" \
  --query "EndpointConfigName" --output text || true)
[[ -n "${ECONFIG_OLD:-}" && "$ECONFIG_OLD" != "None" ]] && echo "Old EndpointConfig: $ECONFIG_OLD"

NEW_EC="serverless-ec-$(date +%s)"
echo "Creating serverless EndpointConfig: $NEW_EC ..."
aws sagemaker create-endpoint-config \
  --endpoint-config-name "$NEW_EC" --region "$REGION" \
  --production-variants "[
    {\"ModelName\":\"$MODEL_NAME\",\"VariantName\":\"$VARIANT_NAME\",
     \"ServerlessConfig\":{\"MemorySizeInMB\":$MEM_MB,\"MaxConcurrency\":$MAX_CONCURRENCY}}
  ]" >/dev/null
echo "Created: $NEW_EC"

EXISTS=$(aws sagemaker list-endpoints --region "$REGION" \
  --query "Endpoints[?EndpointName=='${ENDPOINT}'] | length(@)" --output text)

if [[ "$EXISTS" == "1" ]]; then
  read -r -p "Endpoint '$ENDPOINT' exists and will be DELETED before recreation. Type 'YES' to proceed: " CONFIRM
  [[ "$CONFIRM" == "YES" ]] || { echo "Aborted by user."; exit 1; }
  aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT" --region "$REGION"
fi

aws sagemaker create-endpoint \
  --endpoint-name "$ENDPOINT" \
  --endpoint-config-name "$NEW_EC" \
  --region "$REGION" >/dev/null

echo "Waiting for endpoint to be InService..."
for i in {1..80}; do
  STATUS=$(aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT" --region "$REGION" \
    --query "EndpointStatus" --output text || echo "UNKNOWN")
  echo "  [$i/80] Status: $STATUS"
  [[ "$STATUS" == "InService" ]] && break
  [[ "$STATUS" == "Failed" ]] && { echo "Creation FAILED"; exit 2; }
  sleep 30
done

[[ "$STATUS" == "InService" ]] || { echo "Timeout waiting for InService."; exit 3; }

aws sagemaker-runtime invoke-endpoint \
  --endpoint-name "$ENDPOINT" --region "$REGION" \
  --content-type text/csv --accept application/json \
  --body "$TEST_ROW" out.json >/dev/null || { echo "Invoke failed"; exit 4; }

echo "Response:"; cat out.json; echo
echo "✅ Serverless endpoint is ready: $ENDPOINT"
[[ -n "${ECONFIG_OLD:-}" && "$ECONFIG_OLD" != "None" ]] && \
  echo "Rollback (recreate provisioned): delete endpoint, then:
aws sagemaker create-endpoint --endpoint-name $ENDPOINT --endpoint-config-name $ECONFIG_OLD --region $REGION"
