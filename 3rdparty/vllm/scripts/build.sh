# # update cmake to version >= 3.29
# cp -r cmake-3.29.6 /opt/cmake-3.29.6
# ln -sf /opt/cmake-3.29.6/bin/* /usr/local/bin/

export VLLM_TARGET_DEVICE=rocm
export ROCM_ARCH=gfx928
pip install -e . --no-build-isolation  --index-url https://pypi.org/simple