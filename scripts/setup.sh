# ===============================
# Configuration
# ===============================
if [ -z "$WORKING_DIR" ]; then
    WORKING_DIR=/home/tester/phucnguyen/hygon-test/dev/vllm_plugin
fi

if [ -z "$PYTORCH_ROCM_ARCH" ]; then
    PYTORCH_ROCM_ARCH="gfx928"
fi

VLLM_MOREH_DIR=$WORKING_DIR

rm -rf "$VLLM_MOREH_DIR/build" "$VLLM_MOREH_DIR/vllm_plugin.egg-info" "$VLLM_MOREH_DIR/src/vllm_plugin.egg-info"
pip uninstall -y vllm-plugin

cd "$VLLM_MOREH_DIR" && python3 setup.py develop