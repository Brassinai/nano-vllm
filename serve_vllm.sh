#!/bin/bash
# Launch an OpenAI-compatible vLLM server suitable for Google Colab.
#
# Usage:
#   bash serve_vllm.sh                              # use defaults (Qwen3-0.6B, port 8000)
#   bash serve_vllm.sh --kv-cache-dtype fp8         # quantized KV cache
#   bash serve_vllm.sh --model /path/to/model       # custom model directory
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct bash serve_vllm.sh
#
# The script writes its PID to ./vllm_server.pid and logs to ./vllm_server.log
# so a benchmarking notebook can wait for "Application startup complete." and
# tear the server down deterministically.

set -e


# 1. Environment detection (mirrors setup_colab.sh)

if [ -d "/content" ]; then
    DEFAULT_MODEL_DIR="/content/models/Qwen3-0.6B"
else
    DEFAULT_MODEL_DIR="$HOME/huggingface/Qwen3-0.6B"
fi


# 2. Default arguments

MODEL="${MODEL:-$DEFAULT_MODEL_DIR}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nano-vllm-benchmark}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
LOG_FILE="${LOG_FILE:-vllm_server.log}"
PID_FILE="${PID_FILE:-vllm_server.pid}"
WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
INSTALL_VLLM="${INSTALL_VLLM:-auto}"


# 3. Parse CLI overrides (env vars above are the canonical knobs; flags map
#    onto them so users can pass either style).

print_usage() {
    cat <<'USAGE'
Usage: bash serve_vllm.sh [options]

Options:
  --model PATH                    Local HF model dir or HF repo id
  --served-model-name NAME        Logical name clients use (default: nano-vllm-benchmark)
  --host HOST                     Bind host (default: 127.0.0.1)
  --port PORT                     Bind port (default: 8000)
  --kv-cache-dtype DTYPE          auto | fp8 | fp8_e5m2 | fp8_e4m3 (default: auto)
  --max-model-len N               Context window (default: 4096)
  --max-num-seqs N                Concurrent sequences (default: 256)
  --max-num-batched-tokens N      Scheduler token budget (default: 16384)
  --gpu-memory-utilization FLOAT  Fraction of GPU memory to claim (default: 0.85)
  --tensor-parallel-size N        TP size (default: 1)
  --dtype DTYPE                   Model dtype (default: auto)
  --enforce-eager                 Disable CUDA graphs
  --disable-prefix-caching        Turn prefix caching off (on by default)
  --log-file PATH                 Server log path (default: vllm_server.log)
  --pid-file PATH                 PID file path (default: vllm_server.pid)
  --no-wait                       Do not block until the server is ready
  --ready-timeout SECONDS         How long to wait for readiness (default: 600)
  --install-vllm MODE             auto | always | never (default: auto)
  -h, --help                      Show this help and exit
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER=1; shift ;;
        --disable-prefix-caching) ENABLE_PREFIX_CACHING=0; shift ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        --pid-file) PID_FILE="$2"; shift 2 ;;
        --no-wait) WAIT_FOR_READY=0; shift ;;
        --ready-timeout) READY_TIMEOUT_S="$2"; shift 2 ;;
        --install-vllm) INSTALL_VLLM="$2"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; print_usage; exit 2 ;;
    esac
done


# 4. Install or upgrade vLLM if requested

need_install=0
case "$INSTALL_VLLM" in
    always) need_install=1 ;;
    never)  need_install=0 ;;
    auto)
        if ! python3 -c "import vllm" 2>/dev/null; then
            need_install=1
        fi
        ;;
    *) echo "Invalid --install-vllm mode: $INSTALL_VLLM" >&2; exit 2 ;;
esac

if [ "$need_install" = "1" ]; then
    echo "Installing vLLM..."
    pip install -q --upgrade pip
    pip install -q vllm
fi

VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "vLLM version: $VLLM_VERSION"


# 5. Ensure the model exists if a local path was provided

expand_path() {
    python3 - "$1" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

is_local_path=0
case "$MODEL" in
    /*|./*|../*|~*) is_local_path=1 ;;
esac

if [ "$is_local_path" = "1" ]; then
    MODEL=$(expand_path "$MODEL")
    if [ ! -d "$MODEL" ]; then
        echo "Model directory not found: $MODEL" >&2
        echo "Tip: run setup_colab.sh first, or pass a HuggingFace repo id via --model." >&2
        exit 1
    fi
fi
echo "Model: $MODEL"


# 6. Stop any prior server using the same PID file

if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Stopping previous vLLM server (PID $old_pid)..."
        kill "$old_pid" || true
        for _ in $(seq 1 20); do
            kill -0 "$old_pid" 2>/dev/null || break
            sleep 0.5
        done
        kill -0 "$old_pid" 2>/dev/null && kill -9 "$old_pid" || true
    fi
    rm -f "$PID_FILE"
fi


# 7. Build the launch command

cmd=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --dtype "$DTYPE"
    --disable-log-requests
)
if [ "$ENFORCE_EAGER" = "1" ]; then
    cmd+=(--enforce-eager)
fi
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    cmd+=(--enable-prefix-caching)
else
    cmd+=(--no-enable-prefix-caching)
fi


# 8. Launch in background

: > "$LOG_FILE"
echo "Launching vLLM server: ${cmd[*]}"
echo "  log_file=$LOG_FILE"
echo "  pid_file=$PID_FILE"
nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "Started vLLM server with PID $SERVER_PID"


# 9. Optionally block until the health endpoint responds

if [ "$WAIT_FOR_READY" = "1" ]; then
    echo "Waiting up to ${READY_TIMEOUT_S}s for http://${HOST}:${PORT}/v1/models ..."
    deadline=$((SECONDS + READY_TIMEOUT_S))
    last_print=0
    while [ $SECONDS -lt $deadline ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM process exited before becoming ready. Tail of log:" >&2
            tail -n 80 "$LOG_FILE" >&2 || true
            exit 1
        fi
        if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "vLLM server is ready on http://${HOST}:${PORT}"
            exit 0
        fi
        if [ $((SECONDS - last_print)) -ge 15 ]; then
            echo "  ... still waiting (elapsed ${SECONDS}s)"
            last_print=$SECONDS
        fi
        sleep 2
    done
    echo "Timed out waiting for vLLM to start. Tail of log:" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
fi

echo "Server launched in background; not waiting for readiness."
