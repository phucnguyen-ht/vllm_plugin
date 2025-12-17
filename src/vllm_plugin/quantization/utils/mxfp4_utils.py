import triton
import triton.language as tl
import torch

MXFP_BLOCK_SIZE = tl.constexpr(32)

# fmt: off
@triton.jit
def _upcast_from_mxfp(out_ptr, stride_o_outer, stride_o_quant: tl.constexpr, mx_scale_ptr, stride_scale_outer,
                      stride_scale_quant, mx_tensor_ptr, stride_tensor_outer, stride_tensor_quant: tl.constexpr,
                      outer_dim, quant_dim, BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr):

    tl.static_assert(stride_o_quant == 1, "the weight must be contiguous in the k dimension for mx")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, "BLOCK_SIZE_K must be a multiple of 32")
    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    dst_dtype: tl.constexpr = out_ptr.dtype.element_ty
    tl.static_assert((dst_dtype == tl.float16 or dst_dtype == tl.bfloat16) or dst_dtype == tl.float32)
    tl.static_assert(
        mx_tensor_dtype == tl.uint8
        or ((mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5) or mx_tensor_dtype == dst_dtype),
        "mx_tensor_ptr must be uint8 or float8 or dst_dtype")
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, "mx_scale_ptr must be uint8")

    # Determine if we are dealing with fp8 types.
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    # Compute starting indices for the quantized (packed) dimension and the outer dimension.
    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    start_mxt_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    mx_tensor_ptr += start_mxt_quant * stride_tensor_quant + start_out * stride_tensor_outer
    mx_scale_ptr += start_mx_scale_quant * stride_scale_quant + start_out * stride_scale_outer
    out_ptr += start_out * stride_o_outer + start_out_quant * stride_o_quant

    # Compute offsets and masks.
    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_out_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)
    offs_scale = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)

    mask_outer = start_out + offs_outer < outer_dim
    mask_out_quant = start_out_quant + offs_out_quant < quant_dim
    full_mask_out = mask_out_quant & mask_outer

    mask_src_quant = start_mxt_quant + offs_src_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_src = mask_src_quant & mask_outer

    mask_scale = start_mx_scale_quant + offs_scale < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = mask_scale & mask_outer

    tensor_offsets = offs_src_quant * stride_tensor_quant + offs_outer * stride_tensor_outer
    scale_offsets = offs_scale * stride_scale_quant + offs_outer * stride_scale_outer
    out_offsets = offs_out_quant * stride_o_quant + offs_outer * stride_o_outer

    # Load the packed tensor and scale.
    tensor = tl.load(mx_tensor_ptr + tensor_offsets, mask=full_mask_src)
    scale = tl.load(mx_scale_ptr + scale_offsets, mask=full_scale_mask)

    # Upcast the scale to the destination type.
    if dst_dtype == tl.bfloat16:
        dst_scale = (scale.to(tl.uint16) << 7).to(dst_dtype, bitcast=True)
    else:
        dst_scale = (scale.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        if dst_dtype == tl.float16:
            dst_scale = dst_scale.to(tl.float16)

    # Now upcast the tensor.
    intermediate_dtype: tl.constexpr = tl.bfloat16 if dst_dtype == tl.float32 else dst_dtype
    if is_fp8:
        dst_tensor = tensor.to(intermediate_dtype)
        if tensor.dtype == tl.float8e5:
            from_e_bits: tl.constexpr = 5
            from_m_bits: tl.constexpr = 2
            to_e_bits: tl.constexpr = 8 if intermediate_dtype == tl.bfloat16 else 5
            to_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10

            # Preserve infs and nans. FIXME Fp8E5M2_to_Bf16 doesn't preserve them!
            non_finite_mask_src: tl.constexpr = ((1 << from_e_bits) - 1) << from_m_bits
            non_finite_mask_dst: tl.constexpr = ((1 << to_e_bits) - 1) << to_m_bits
            dst_tensor = tl.where(
                (tensor.to(tl.uint8, bitcast=True) & non_finite_mask_src) == non_finite_mask_src,
                (dst_tensor.to(tl.uint16, bitcast=True) | non_finite_mask_dst).to(intermediate_dtype, bitcast=True),
                dst_tensor,
            )
    else:
        assert is_fp4
        dst_bias: tl.constexpr = 127 if intermediate_dtype == tl.bfloat16 else 15
        dst_0p5: tl.constexpr = 16128 if intermediate_dtype == tl.bfloat16 else 0x3800
        dst_m_bits: tl.constexpr = 7 if intermediate_dtype == tl.bfloat16 else 10
        # e2m1
        em0 = tensor & 0x07
        em1 = tensor & 0x70
        x0 = (em0.to(tl.uint16) << (dst_m_bits - 1)) | ((tensor & 0x08).to(tl.uint16) << 12)
        x1 = (em1.to(tl.uint16) << (dst_m_bits - 5)) | ((tensor & 0x80).to(tl.uint16) << 8)
        # Three cases:
        # 1) x is normal and non-zero: Correct bias
        x0 = tl.where((em0 & 0x06) != 0, x0 + ((dst_bias - 1) << dst_m_bits), x0)
        x1 = tl.where((em1 & 0x60) != 0, x1 + ((dst_bias - 1) << dst_m_bits), x1)
        # 2) x is subnormal (x == 0bs001 where s is the sign): Map to +-0.5 in the dst type
        x0 = tl.where(em0 == 0x01, dst_0p5 | (x0 & 0x8000), x0)
        x1 = tl.where(em1 == 0x10, dst_0p5 | (x1 & 0x8000), x1)
        # 3) x is zero, do nothing
        dst_tensor = tl.interleave(x0, x1).to(intermediate_dtype, bitcast=True)
    dst_tensor = dst_tensor.to(dst_dtype)

    # Reshape for proper broadcasting: the scale was stored with a 32‐sized “inner” grouping.
    dst_tensor = dst_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    dst_scale = dst_scale.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 1])
    scale = scale.reshape(dst_scale.shape)

    out_tensor = dst_tensor * dst_scale
    # Correct any NaNs encoded via the scale.
    out_tensor = tl.where(scale == 0xFF, float("nan"), out_tensor)
    out_tensor = out_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    tl.store(out_ptr + out_offsets, out_tensor, mask=full_mask_out)
    
def upcast_from_mxfp(tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype, axis: int):
    """
    Upcasts an mxfp (packed) weight tensor back to float16 or bfloat16.

    The function assumes that the tensors were quantized along the given axis.
    It permutes the tensor so that the quantized axis is last, reshapes to 2D,
    launches the Triton upcast kernel, and then unpermutes back to the original order.
    """
    ndim = tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    assert tensor.ndim == scale.ndim, (f"Weight and scale must have the same number of dimensions. "
                                       f"Got {tensor.ndim=} and {scale.ndim=}")
    # dtype checks
    assert tensor.dtype in {torch.uint8, torch.float8_e5m2, torch.float8_e4m3fn}, \
        f"Invalid tensor dtype {tensor.dtype=}"
    assert scale.dtype == torch.uint8, f"Invalid scale dtype {scale.dtype=}"
    assert target_dtype in (torch.float16, torch.bfloat16, torch.float32), f"Invalid output dtype {target_dtype=}"
    # upcast
    logical_quant_dim = tensor.shape[axis] * (2 if tensor.dtype == torch.uint8 else 1)
    tensor = tensor.transpose(axis, tensor.ndim - 1).contiguous()
    scale = scale.transpose(axis, scale.ndim - 1).contiguous()
    out = torch.empty((*tensor.shape[:-1], logical_quant_dim), dtype=target_dtype, device=tensor.device)
    reshaped_out = out.view(-1, out.shape[-1])
    reshaped_tensor = tensor.view(-1, tensor.shape[-1])
    reshaped_scale = scale.view(-1, scale.shape[-1])
    BLOCK_OUT_DIM = 128
    BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
    blocks_out_dim = triton.cdiv(reshaped_out.shape[0], BLOCK_OUT_DIM)
    blocks_quant_dim = triton.cdiv(reshaped_out.shape[1], BLOCK_QUANT_DIM)
    _upcast_from_mxfp[(blocks_out_dim, blocks_quant_dim)](reshaped_out, *reshaped_out.stride(), reshaped_scale,
                                                          *reshaped_scale.stride(), reshaped_tensor,
                                                          *reshaped_tensor.stride(), *reshaped_out.shape, BLOCK_OUT_DIM,
                                                          BLOCK_QUANT_DIM, num_warps=8)
    out = out.transpose(axis, scale.ndim - 1).contiguous()
    return out

from enum import Enum
class DequantScaleRoundingMode(Enum):
    ROUND_UP = 0
    ROUND_DOWN = 1

@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")

@triton.jit
def _compute_quant_and_scale(src_tensor, valid_src_mask, mx_tensor_dtype: tl.constexpr,
                             DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0):
    is_fp8: tl.constexpr = mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // MXFP_BLOCK_SIZE

    # Explicit cast to fp32 since most ops are not supported on bfloat16. We avoid needless conversions to and from bf16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(valid_src_mask, abs_tensor, -1.0)  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)
    dequant_scale = max_val / _get_max_quant_val(mx_tensor_dtype)
    if DEQUANT_SCALE_ROUNDING_MODE == 0:
        # DequantScaleRoundingMode.ROUND_UP
        # compute 2 ** ceil(log2(dequant_scale))
        # Adding 0x007FFFFF adds exponent by 1 unless mantissa is all zeros
        # A corner case: exponent is 0xFF that will overflow but that's already
        # NaN so assume we don't care.
        dequant_scale_exponent = (dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    else:
        # DequantScaleRoundingMode.ROUND_DOWN
        # compute 2 ** floor(log2(dequant_scale))
        assert DEQUANT_SCALE_ROUNDING_MODE == 1
        dequant_scale_exponent = dequant_scale.to(tl.uint32, bitcast=True) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = tl.reshape(f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, MXFP_BLOCK_SIZE])
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0. This will ensure that any padding tensors are 0 in the mx format.
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE])

    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)
    # Now we must convert the tensors to the mx format.
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = (quant_tensor & 0x7FFFFF)

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(E8_BIAS, exponents + 1, sanitize_overflow=False)
        mantissas = tl.where(exponents < E8_BIAS, (0x400000 | (mantissas >> 1)) >> adjusted_exponents, mantissas)

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2])
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent

@triton.jit
def _downcast_to_mxfp(mx_tensor_ptr, stride_mxt_outer, stride_mxt_quant: tl.constexpr,
                      mx_scale_ptr, stride_mx_scale_outer, stride_mx_scale_quant,
                      src_ptr, stride_src_outer, stride_src_quant,
                      outer_dim, quant_dim,
                      BLOCK_SIZE_OUT_DIM: tl.constexpr, BLOCK_SIZE_QUANT_DIM: tl.constexpr,
                      DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr):

    tl.static_assert(stride_mxt_quant == 1, f"Output stride, {stride_mxt_quant=} must be 1.")
    tl.static_assert(BLOCK_SIZE_QUANT_DIM % MXFP_BLOCK_SIZE == 0, f"{BLOCK_SIZE_QUANT_DIM=} must be a multiple of 32")

    # uint8 signifies two fp4 e2m1 values packed into a single byte
    mx_tensor_dtype: tl.constexpr = mx_tensor_ptr.dtype.element_ty
    tl.static_assert(mx_tensor_dtype == tl.uint8 or (mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5),
                     f"Invalid {mx_tensor_dtype=}. Must be uint8 or float8.")

    src_dtype: tl.constexpr = src_ptr.dtype.element_ty
    tl.static_assert(mx_scale_ptr.dtype.element_ty == tl.uint8, f"{mx_scale_ptr.dtype.element_ty=} must be uint8")
    tl.static_assert(((src_dtype == tl.bfloat16) or (src_dtype == tl.float16)) or (src_dtype == tl.float32), f"{src_dtype=} must be bfloat16 or float16 or float32")
    is_fp4: tl.constexpr = mx_tensor_dtype == tl.uint8

    outer_block = tl.program_id(0).to(tl.int64)
    quant_block = tl.program_id(1).to(tl.int64)

    K_DIVISOR: tl.constexpr = 2 if is_fp4 else 1
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = BLOCK_SIZE_QUANT_DIM // MXFP_BLOCK_SIZE
    BLOCK_SIZE_QUANT_MX_TENSOR: tl.constexpr = BLOCK_SIZE_QUANT_DIM // K_DIVISOR

    start_src_quant = quant_block * BLOCK_SIZE_QUANT_DIM
    start_mx_scale_quant = quant_block * BLOCK_SIZE_QUANT_MX_SCALE
    start_mx_quant = quant_block * BLOCK_SIZE_QUANT_MX_TENSOR
    start_out = outer_block * BLOCK_SIZE_OUT_DIM

    src_ptr += start_src_quant * stride_src_quant + start_out * stride_src_outer
    mx_scale_ptr += start_mx_scale_quant * stride_mx_scale_quant + start_out * stride_mx_scale_outer
    mx_tensor_ptr += start_mx_quant * stride_mxt_quant + start_out * stride_mxt_outer

    offs_src_quant = tl.arange(0, BLOCK_SIZE_QUANT_DIM)[None, :].to(tl.int64)
    offs_mxt_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_TENSOR)[None, :].to(tl.int64)
    offs_scale_quant = tl.arange(0, BLOCK_SIZE_QUANT_MX_SCALE)[None, :].to(tl.int64)
    offs_outer = tl.arange(0, BLOCK_SIZE_OUT_DIM)[:, None].to(tl.int64)

    mask_src_quant = start_src_quant + offs_src_quant < quant_dim
    mask_n = start_out + offs_outer < outer_dim
    full_mask_src = mask_src_quant & mask_n

    mask_mxt_quant = start_mx_quant + offs_mxt_quant < tl.cdiv(quant_dim, K_DIVISOR)
    full_mask_mxt = mask_mxt_quant & mask_n

    scale_mask_k = start_mx_scale_quant + offs_scale_quant < tl.cdiv(quant_dim, MXFP_BLOCK_SIZE)
    full_scale_mask = scale_mask_k & mask_n

    src_tensor_offsets = offs_src_quant * stride_src_quant + offs_outer * stride_src_outer
    mx_scale_offsets = offs_scale_quant * stride_mx_scale_quant + offs_outer * stride_mx_scale_outer
    mx_tensor_offsets = offs_mxt_quant * stride_mxt_quant + offs_outer * stride_mxt_outer
    src_tensor = tl.load(src_ptr + src_tensor_offsets, mask=full_mask_src)

    out_tensor, scale_tensor = _compute_quant_and_scale(src_tensor, full_mask_src, mx_tensor_dtype,
                                                        DEQUANT_SCALE_ROUNDING_MODE)

    tl.store(mx_scale_ptr + mx_scale_offsets, scale_tensor, mask=full_scale_mask)
    tl.store(mx_tensor_ptr + mx_tensor_offsets, out_tensor, mask=full_mask_mxt)


def downcast_to_mxfp(src_tensor: torch.Tensor, out_quant_type: torch.dtype, axis: int,
                     DEQUANT_SCALE_ROUNDING_MODE: DequantScaleRoundingMode = DequantScaleRoundingMode.ROUND_UP):
    """
         Convert the src weights to mx format. The src weight is quantized along the axis dimension.

         If weight_quant_type is torch.uint8, we output mxfp4 where two e2m1 values are packed into a single byte.
         Note that this means the k_dim of the tensor will be half of the logical k_dim.

         If weight_quant_type is torch.float8_e4m3fn or torch.float8_e5m2, we output mxfp8 with the float8s are stored
         in their respective formats.
    """
    ndim = src_tensor.ndim
    assert -ndim <= axis < ndim, f"Invalid axis {axis=}"
    axis = axis if axis >= 0 else axis + ndim
    # downcast
    src_tensor = src_tensor.transpose(axis, src_tensor.ndim - 1)
    is_fp4 = out_quant_type == torch.uint8
    is_fp8 = out_quant_type in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert is_fp4 or is_fp8
    divisor = 2 if is_fp4 else 1
    L = src_tensor.shape[-1]
    if is_fp4:
        assert L % 2 == 0, f"axis dim must be divisible by 2 for e2m1. Got {L}"
    out_shape = src_tensor.shape[:-1] + (L // divisor, )
    out_scale_shape = src_tensor.shape[:-1] + (triton.cdiv(L, MXFP_BLOCK_SIZE), )

    out_quant_tensor = src_tensor.new_empty(out_shape, dtype=out_quant_type)
    out_scale = src_tensor.new_empty(out_scale_shape, dtype=torch.uint8)

    if src_tensor.numel() > 0:
        kernel_src_tensor = src_tensor.reshape(-1, src_tensor.shape[-1])
        kernel_quant_tensor = out_quant_tensor.view(-1, out_quant_tensor.shape[-1])
        kernel_scale = out_scale.view(-1, out_scale.shape[-1])

        BLOCK_OUT_DIM = 128
        BLOCK_QUANT_DIM = MXFP_BLOCK_SIZE.value
        grid_out = triton.cdiv(kernel_src_tensor.shape[0], BLOCK_OUT_DIM)
        grid_quant = triton.cdiv(kernel_src_tensor.shape[1], BLOCK_QUANT_DIM)

        _downcast_to_mxfp[(grid_out, grid_quant)](kernel_quant_tensor, *kernel_quant_tensor.stride(), kernel_scale,
                                                *kernel_scale.stride(), kernel_src_tensor, *kernel_src_tensor.stride(),
                                                *kernel_src_tensor.shape, BLOCK_OUT_DIM, BLOCK_QUANT_DIM,
                                                DEQUANT_SCALE_ROUNDING_MODE.value, num_warps=8)

    out_quant_tensor = out_quant_tensor.transpose(axis, src_tensor.ndim - 1)
    out_scale = out_scale.transpose(axis, src_tensor.ndim - 1)
    return out_quant_tensor, out_scale