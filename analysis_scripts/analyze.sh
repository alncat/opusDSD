if [[ $# -ge 5 && "$5" != --* ]]; then
    PSAMPLE="$5"
    EXTRA_ARGS=("${@:6}")
else
    PSAMPLE="10"
    EXTRA_ARGS=("${@:5}")
fi

dsd analyze "$1" "$2" --D 192 --pose "$1/pose.$2.pkl" --pc "$3" --ksample "$4" --psample "$PSAMPLE" "${EXTRA_ARGS[@]}"
