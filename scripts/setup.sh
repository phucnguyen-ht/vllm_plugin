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

# # build cmake to install fused_moe
# if [ ! -d "$VLLM_MOREH_DIR/3rdparty/cmake-3.29.6" ]; then
#     cd "$VLLM_MOREH_DIR/3rdparty"
#     wget https://github.com/Kitware/CMake/releases/download/v3.29.6/cmake-3.29.6.tar.gz
#     tar -xvf cmake-3.29.6.tar.gz
#     cd cmake-3.29.6

#     ./bootstrap --prefix=/opt/cmake-3.29.6
#     make -j$(nproc)
#     make install
#     ln -s /opt/cmake-3.29.6/bin/cmake /usr/bin/cmake
# else
#     cd "$VLLM_MOREH_DIR/3rdparty/cmake-3.29.6"
#     make install
#     ln -s /opt/cmake-3.29.6/bin/cmake /usr/bin/cmake
# fi

# install fused_moe
cd "$VLLM_MOREH_DIR/3rdparty/fused_moe" && python3 setup_hygon.py develop

# replace aiter import error => unable to use aiter because of HIP Device Function
AITER_PACKAGE_DIR="/usr/local/lib/python3.10/dist-packages/aiter"
cp "$VLLM_MOREH_DIR/3rdparty/aiter/utility/dtypes.py" "$AITER_PACKAGE_DIR/utility/dtypes.py"

# install vllm-plugin
rm -rf "$VLLM_MOREH_DIR/build" "$VLLM_MOREH_DIR/vllm_plugin.egg-info" "$VLLM_MOREH_DIR/src/vllm_plugin.egg-info"
pip uninstall -y vllm-plugin
cd "$VLLM_MOREH_DIR" && python3 setup.py develop