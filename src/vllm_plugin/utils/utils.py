# import torch
# import re

# COUNT = 0
# def print_debug(a: torch.Tensor, file_path: str, name: str, print_shape=False, dash=0, layer=-1):
#     global COUNT
    
#     def print_dash(f, dash=0):
#         if dash > 0:
#             f.write("-" * dash + "\n")

#     if a.shape[0] not in [1, 10]: return
#     if layer==0: COUNT += 1
#     if COUNT>=5: return
    
    
#     P = 256
#     MAX_SLICE = 16

#     if a.ndim > 3:
#         raise ValueError(f"Tensor '{name}' has {a.ndim} dims. Max 3 supported.")

#     a_cpu = a.detach().cpu()
#     a_print = a_cpu

#     if any(dim > P for dim in a_cpu.shape):
#         slices = tuple(slice(0, MAX_SLICE) if dim > P else slice(None) for dim in a_cpu.shape)
#         a_print = a_cpu[slices]

#     try:
#         torch.set_printoptions(threshold=float("inf"), linewidth=20000, sci_mode=False, precision=4)
#         tensor_str = str(a_print)
#         tensor_str = re.sub(r"^tensor\(", "", tensor_str)
#         tensor_str = re.sub(r",\s*(dtype|device)=.*?\)$", "", tensor_str, flags=re.DOTALL)
#         tensor_str = re.sub(r"\)$", "", tensor_str)
#     except Exception as e:
#         raise RuntimeError("Tensor print failed") from e

#     with open(file_path, "a", encoding="utf-8") as f:
        
#         f.write(
#             f"name={name!r:<25} | "
#             f"dtype={str(a.dtype):<15} | "
#             f"shape={str(tuple(a.shape)):<20} | "
#             f"{('val = ' + str(a_cpu.item()) if a_cpu.numel() == 1 else '')}\n"
#         )
#         if not print_shape:
#             f.write(f"{tensor_str}\n")
#         print_dash(f, dash)