#!/usr/bin/env bash
set -euo pipefail

# ---- defaults ----
INPUT_FILE=""
MODEL_FILE=""
NUM_CLUSTERS=""
NBC=""
BCSIZE=""
GRAPH_THRESH=""
RUN_REMOVE_DOMINATORS=true
RUN_REMOVE_DOMINATORS_ON_CLUSTER=true
DELETE_EM_FILES_CLUSTER=true

# Optional: root directory of your project
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#echo "$(pwd)";

usage() {
    echo "Usage:"
    echo "  $0 --input_file FILE --model_file FILE --num_clusters N --nbc N --bcsize N"
    echo
    echo "Example:"
    echo "  $0 \\"
    echo "    --input_file inputsup/PLAIN_small_07.rmp \\"
    echo "    --model_file PLAIN_small_07_epoch_1000.pt \\"
    echo "    --num_clusters 2 \\"
    echo "    --nbc 30 \\"
    echo "    --bcsize 100 \\"
    echo "    --graph_thresh 1000 \\"
    echo "    --run_removes_dominator true/false \\"
    echo "    --run_removes_dominator_on_cluster true/false \\"
    echo "    --delete_em_Files_cluster true/false \\"
    exit 1
}

# ---- parse named args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_file)
            [[ -n "${2:-}" ]] || { echo "Missing value for --input_file"; exit 1; }
            INPUT_FILE="$2"
            shift 2
            ;;
        --model_file)
            [[ -n "${2:-}" ]] || { echo "Missing value for --model_file"; exit 1; }
            MODEL_FILE="$2"
            shift 2
            ;;
        --num_clusters)
            [[ -n "${2:-}" ]] || { echo "Missing value for --num_clusters"; exit 1; }
            NUM_CLUSTERS="$2"
            shift 2
            ;;
        --nbc)
            [[ -n "${2:-}" ]] || { echo "Missing value for --nbc"; exit 1; }
            NBC="$2"
            shift 2
            ;;
        --bcsize)
            [[ -n "${2:-}" ]] || { echo "Missing value for --bcsize"; exit 1; }
            BCSIZE="$2"
            shift 2
            ;;
        --graph_thresh)
            [[ -n "${2:-}" ]] || { echo "Missing value for --graph_thresh"; exit 1; }
            GRAPH_THRESH="$2"
            shift 2
            ;;
        --run_remove_dominator)
            [[ -n "${2:-}" ]] || { echo "Missing value for --run_remove_dominators"; exit 1; }
            RUN_REMOVE_DOMINATORS="$2"
            shift 2
            ;;
        --run_remove_dominator_on_cluster)
            [[ -n "${2:-}" ]] || { echo "Missing value for --run_remove_dominators_on_cluster"; exit 1; }
            RUN_REMOVE_DOMINATORS_ON_CLUSTER="$2"
            shift 2
            ;;
        --delete_em_files_cluster)
            [[ -n "${2:-}" ]] || { echo "Missing value for --delete_em_files_cluster"; exit 1; }
            DELETE_EM_FILES_CLUSTER="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# ---- validate required args ----
[[ -n "$INPUT_FILE" ]] || { echo "Error: --input_file is required"; usage; }
[[ -n "$MODEL_FILE" ]] || { echo "Error: --model_file is required"; usage; }
[[ -n "$NUM_CLUSTERS" ]] || { echo "Error: --num_clusters is required"; usage; }
[[ -n "$NBC" ]] || { echo "Error: --nbc is required"; usage; }
[[ -n "$BCSIZE" ]] || { echo "Error: --bcsize is required"; usage; }
[[ -n "$GRAPH_THRESH" ]] || { echo "Error: --graph_thresh is required"; usage; }

echo "---------------------------------------------------------------------------"
echo "Project root:                        $PROJECT_ROOT"
echo "Input file:                          $INPUT_FILE"
echo "Model file:                          $MODEL_FILE"
echo "Num clusters:                        $NUM_CLUSTERS"
echo "NBC:                                 $NBC"
echo "BC size:                             $BCSIZE"
echo "Graph Threshold size:                $GRAPH_THRESH"
echo "Run Remove dominator:                $RUN_REMOVE_DOMINATORS"
echo "Run Remove dominator on cluster:     $RUN_REMOVE_DOMINATORS_ON_CLUSTER"
echo "Delete em files after clustering:    $DELETE_EM_FILES_CLUSTER"
echo "---------------------------------------------------------------------------"
# --------------------------------------------------
# Step 1: Remove em files
# --------------------------------------------------
#pushd "$PROJECT_ROOT/.." > /dev/null

if [[ "$RUN_REMOVE_DOMINATORS" == "true" ]]; then
    EM_FILES=( "${INPUT_FILE}-em*" )

#    if (( ${#EM_FILES[@]} > 0 )); then
    if compgen -G "${INPUT_FILE}-*" > /dev/null; then
        echo "EM files already exist for ${EM_FILES} ${INPUT_FILE}, skipping removedominators.py"
    else
        echo "No EM files found, running removedominators.py ..."
        echo "cleaning old em files for $INPUT_FILE-*"
        rm -f "$INPUT_FILE"-* || true
        # ------------------------------------
        #  Step 2: Run removedominators.py
        # ------------------------------------
        echo "Running removedominators.py ..."
        python3 removedominators.py "$INPUT_FILE"
    fi
else
    echo "Skipping running removedominators.py step"
fi

#popd > /dev/null

# --------------------------------------------------
# Step 3: go into community_detection and run detect_communities.py
# --------------------------------------------------
pushd "$PROJECT_ROOT/community_detection" > /dev/null
echo "Remove cluster files"
rm -f "${INPUT_FILE}_cluster*$" || true


echo "Running detect_communities.py ..."
python3 detect_communities.py \
    --files "../$INPUT_FILE" \
    --model_file "$MODEL_FILE" \
    --num_clusters "$NUM_CLUSTERS" \
    --nbc "$NBC" \
    --bcsize "$BCSIZE" \
    --graph_thresh "$GRAPH_THRESH" \
    --run_remove_dominators "$RUN_REMOVE_DOMINATORS" \
    --run_remove_dominators_on_cluster "$RUN_REMOVE_DOMINATORS_ON_CLUSTER" \
    --delete_em_files "$DELETE_EM_FILES_CLUSTER"

# ----------------------------------------
# Step 4: Remove cluster files
# ----------------------------------------
if [[ "$DELETE_EM_FILES_CLUSTER" == "true" ]]; then
    echo "Remove cluster files"
    #rm -f "${INPUT_FILE}_cluster*" || true
fi
popd > /dev/null

#echo "Remove em files for $INPUT_FILE-*"
#rm -f "$INPUT_FILE"-* || true

echo "Done."
