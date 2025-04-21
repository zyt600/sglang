import torch


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 4,
    num_warps_per_block: int = 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    src_layer_offset: int,
    dst_layer_offset: int,
    block_quota: int = 4,
    num_warps_per_block: int = 16,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        src_layer_offset,
        dst_layer_offset,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 4,
    num_warps_per_block: int = 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    src_layer_offset: int,
    dst_layer_offset: int,
    block_quota: int = 4,
    num_warps_per_block: int = 16,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        num_layers,
        src_layer_offset,
        dst_layer_offset,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_to_cpu_all_layer_direct(
    host_indices: torch.Tensor,
    host_k_buffer: torch.Tensor,
    host_v_buffer: torch.Tensor,
    device_indices: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    page_size: int,
    layer_num: int,
):
    torch.ops.sgl_kernel.transfer_kv_to_cpu_all_layer_direct(
        host_indices,
        host_k_buffer,
        host_v_buffer,
        device_indices,
        device_k_buffer,
        device_v_buffer,
        page_size,
        layer_num,
    )


def transfer_kv_to_gpu_per_layer_direct(
    host_indices: torch.Tensor,
    host_k_buffer: torch.Tensor,
    host_v_buffer: torch.Tensor,
    device_indices: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    page_size: int,
    layer_id: int,
):
    torch.ops.sgl_kernel.transfer_kv_to_gpu_per_layer_direct(
        host_indices,
        host_k_buffer,
        host_v_buffer,
        device_indices,
        device_k_buffer,
        device_v_buffer,
        page_size,
        layer_id,
    )


def transfer_kv_to_cpu_all_layer_direct_mla(
    host_indices: torch.Tensor,
    host_buffer: torch.Tensor,
    device_indices: torch.Tensor,
    device_buffer: torch.Tensor,
    page_size: int,
    layer_num: int,
):
    torch.ops.sgl_kernel.transfer_kv_to_cpu_all_layer_direct_mla(
        host_indices,
        host_buffer,
        device_indices,
        device_buffer,
        page_size,
        layer_num,
    )


def transfer_kv_to_gpu_per_layer_direct_mla(
    host_indices: torch.Tensor,
    host_buffer: torch.Tensor,
    device_indices: torch.Tensor,
    device_buffer: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_to_gpu_per_layer_direct_mla(
        host_indices,
        host_buffer,
        device_indices,
        device_buffer,
        page_size,
    )
