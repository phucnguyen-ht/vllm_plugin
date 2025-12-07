import torch
from vllm.attention.backends.flashmla import FlashMLAMetadata
from vllm.attention.backends.mla.common import MLACommonMetadata
from vllm.attention.backends.rocm_flash_attn import ROCmFlashAttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.utils import async_tensor_h2d

def cumsum(lst):
    cum_lst = [0]
    sum = 0
    for i in range(0, len(lst)):
        sum = sum + lst[i]
        cum_lst.append(sum)
    return cum_lst

def is_supported_attention_metadata(atten_metadata):
    return isinstance(atten_metadata, ROCmFlashAttentionMetadata) or \
            isinstance(atten_metadata, FlashMLAMetadata) or \
            isinstance(atten_metadata, MLACommonMetadata) or \
            isinstance(atten_metadata, FlashAttentionMetadata)

def split_model_input(model_input, self_device, batch_size_left, batch_size_right):
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
    query_tokens_split = [sum(model_input.query_lens[0:batch_size_left]), sum(model_input.query_lens[batch_size_left:])]
    batch_size_split = [batch_size_left, batch_size_right]
    split_input_tokens = torch.split(model_input.input_tokens, query_tokens_split, dim=0)
    split_input_positions = torch.split(model_input.input_positions, query_tokens_split, dim=0)
    seq_lens_left = model_input.attn_metadata.seq_lens[0:batch_size_left]
    seq_lens_right = model_input.attn_metadata.seq_lens[batch_size_left:]
    query_lens_left = model_input.query_lens[0:batch_size_left]
    query_lens_right = model_input.query_lens[batch_size_left:]
    split_seq_lens_tensor = torch.split(model_input.attn_metadata.seq_lens_tensor, batch_size_split, dim=0)
    split_block_tables = torch.split(model_input.attn_metadata.block_tables, batch_size_split, dim=0)
    num_prefills_left = 0
    num_prefills_right = 0
    num_prefill_tokens_left = 0
    num_prefill_tokens_right = 0
    num_decode_tokens_left = 0
    num_decode_tokens_right = 0
    max_prefill_seq_len_left = 0
    max_prefill_seq_len_right = 0
    max_decode_seq_len_left = 0
    max_decode_seq_len_right = 0
    max_decode_query_len_left = None
    max_decode_query_len_right = None
    encoder_seq_lens_left = None
    encoder_seq_lens_right = None
    encoder_seq_lens_tensor_left = None
    encoder_seq_lens_tensor_right = None
    max_encoder_seq_len_left = None
    max_encoder_seq_len_right = None
    num_encoder_tokens_left = None
    num_encoder_tokens_right = None
    cross_slot_mapping_left = None
    cross_slot_mapping_right = None
    cross_block_tables_left = None
    cross_block_tables_right = None
    if model_input.is_prompt:
        num_prefills_left = batch_size_left
        num_prefills_right = batch_size_right
        num_prefill_tokens_left = sum(model_input.query_lens[0:batch_size_left])
        num_prefill_tokens_right = sum(model_input.query_lens[batch_size_left:])
        max_prefill_seq_len_left = max(model_input.attn_metadata.seq_lens[0:batch_size_left])
        max_prefill_seq_len_right = max(model_input.attn_metadata.seq_lens[batch_size_left:])
    else:
        num_decode_tokens_left = batch_size_left
        num_decode_tokens_right = batch_size_right
        max_decode_seq_len_left = max(model_input.attn_metadata.seq_lens[0:batch_size_left])
        max_decode_seq_len_right = max(model_input.attn_metadata.seq_lens[batch_size_left:])
    split_slot_mapping = torch.split(model_input.attn_metadata.slot_mapping, query_tokens_split, dim=0)
    max_query_len_left = max(model_input.query_lens[0:batch_size_left])
    max_query_len_right = max(model_input.query_lens[batch_size_left:])
    zero_tensor = torch.tensor([0], device=self_device, dtype=torch.int32)
    query_start_loc_left_list = cumsum(query_lens_left)
    query_start_loc_right_list = cumsum(query_lens_right)
    query_start_loc_left = async_tensor_h2d(query_start_loc_left_list, torch.int32,
                                            self_device,
                                            True)
    query_start_loc_right = async_tensor_h2d(query_start_loc_right_list, torch.int32,
                                            self_device,
                                            True)
    seq_start_loc_left = torch.cat((zero_tensor, split_seq_lens_tensor[0].cumsum(dim=0)), dim=0).to(torch.int32)
    seq_start_loc_right = torch.cat((zero_tensor, split_seq_lens_tensor[1].cumsum(dim=0)), dim=0).to(torch.int32)

    split_context_lens_tensor = torch.split(model_input.attn_metadata.context_lens_tensor, batch_size_split, dim=0)
    request_ids_to_seq_ids_left = {}
    request_ids_to_seq_ids_right = {}
    counter = 0
    for key, value in model_input.request_ids_to_seq_ids.items():
        if counter < batch_size_left:
            request_ids_to_seq_ids_left[key] = value
        else:
            request_ids_to_seq_ids_right[key] = value
        counter += 1

    previous_hidden_states_left = None
    previous_hidden_states_right = None
    if model_input.previous_hidden_states != None:
        split_previous_hidden_states = torch.split(model_input.previous_hidden_states, query_tokens_split, dim=0)
        previous_hidden_states_left = split_previous_hidden_states[0]
        previous_hidden_states_right = split_previous_hidden_states[1]    
    
    if isinstance(model_input.attn_metadata, MLACommonMetadata):
        attn_metadata_left = MLACommonMetadata(
            num_prefills = num_prefills_left,
            num_prefill_tokens = num_prefill_tokens_left,
            num_decode_tokens = num_decode_tokens_left,
            slot_mapping = split_slot_mapping[0],
            multi_modal_placeholder_index_maps = model_input.attn_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            input_positions = split_input_positions[0],
            seq_lens = seq_lens_left,
            seq_lens_tensor = split_seq_lens_tensor[0],
            max_prefill_seq_len = max_prefill_seq_len_left,
            max_decode_seq_len = max_decode_seq_len_left,
            context_lens_tensor = split_context_lens_tensor[0],
            block_tables = split_block_tables[0],
            max_query_len = max_query_len_left,
            max_decode_query_len = max_decode_query_len_left,
            query_start_loc = query_start_loc_left,
            seq_start_loc = seq_start_loc_left,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            head_dim = model_input.attn_metadata.head_dim,
            is_profile_run = model_input.attn_metadata.is_profile_run,
            context_chunk_cu_seq_lens=model_input.attn_metadata.context_chunk_cu_seq_lens, 
            context_chunk_starts=model_input.attn_metadata.context_chunk_starts, 
            context_chunk_seq_tot=model_input.attn_metadata.context_chunk_seq_tot, 
            context_chunk_max_seq_lens=model_input.attn_metadata.context_chunk_max_seq_lens, 
            context_chunk_workspace=model_input.attn_metadata.context_chunk_workspace, 
        )
        attn_metadata_right = MLACommonMetadata(
            num_prefills = num_prefills_right,
            num_prefill_tokens = num_prefill_tokens_right,
            num_decode_tokens = num_decode_tokens_right,
            slot_mapping = split_slot_mapping[1],
            multi_modal_placeholder_index_maps = model_input.attn_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            input_positions = split_input_positions[1],
            seq_lens = seq_lens_right,
            seq_lens_tensor = split_seq_lens_tensor[1],
            max_prefill_seq_len = max_prefill_seq_len_right,
            max_decode_seq_len = max_decode_seq_len_right,
            context_lens_tensor = split_context_lens_tensor[1],
            block_tables = split_block_tables[1],
            max_query_len = max_query_len_right,
            max_decode_query_len = max_decode_query_len_right,
            query_start_loc = query_start_loc_right,
            seq_start_loc = seq_start_loc_right,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            head_dim = model_input.attn_metadata.head_dim,
            is_profile_run = model_input.attn_metadata.is_profile_run,
            context_chunk_cu_seq_lens=model_input.attn_metadata.context_chunk_cu_seq_lens, 
            context_chunk_starts=model_input.attn_metadata.context_chunk_starts, 
            context_chunk_seq_tot=model_input.attn_metadata.context_chunk_seq_tot, 
            context_chunk_max_seq_lens=model_input.attn_metadata.context_chunk_max_seq_lens, 
            context_chunk_workspace=model_input.attn_metadata.context_chunk_workspace, 
        )
    
    if isinstance(model_input.attn_metadata, ROCmFlashAttentionMetadata):
        block_tables_list_left = model_input.attn_metadata.block_tables_list[0:batch_size_left]
        block_tables_list_right = model_input.attn_metadata.block_tables_list[batch_size_left:]
        attn_metadata_left = ROCmFlashAttentionMetadata(
            seq_lens_tensor = split_seq_lens_tensor[0],
            max_decode_seq_len = max_decode_seq_len_left,
            block_tables = split_block_tables[0],
            num_prefills = num_prefills_left,
            num_prefill_tokens = num_prefill_tokens_left,
            num_decode_tokens = num_decode_tokens_left,
            slot_mapping = split_slot_mapping[0],
            multi_modal_placeholder_index_maps = {},
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            seq_lens = seq_lens_left,
            max_prefill_seq_len = max_prefill_seq_len_left,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            max_query_len = max_query_len_left,
            query_start_loc = query_start_loc_left,
            seq_start_loc = seq_start_loc_left,
            context_lens_tensor = split_context_lens_tensor[0],
            max_decode_query_len = max_decode_query_len_left,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            tree_attention_masks_tensor = None,
            block_tables_list = block_tables_list_left,
            encoder_seq_lens = encoder_seq_lens_left,
            encoder_seq_lens_tensor = encoder_seq_lens_tensor_left,
            max_encoder_seq_len = max_encoder_seq_len_left,
            num_encoder_tokens = num_encoder_tokens_left,
            cross_slot_mapping = cross_slot_mapping_left,
            cross_block_tables = cross_block_tables_left,
        )
        attn_metadata_right = ROCmFlashAttentionMetadata(
            seq_lens_tensor = split_seq_lens_tensor[1],
            max_decode_seq_len = max_decode_seq_len_right,
            block_tables = split_block_tables[1],
            num_prefills = num_prefills_right,
            num_prefill_tokens = num_prefill_tokens_right,
            num_decode_tokens = num_decode_tokens_right,
            slot_mapping = split_slot_mapping[1],
            multi_modal_placeholder_index_maps = {},
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            seq_lens = seq_lens_right,
            max_prefill_seq_len = max_prefill_seq_len_right,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            max_query_len = max_query_len_right,
            query_start_loc = query_start_loc_right,
            seq_start_loc = seq_start_loc_right,
            context_lens_tensor = split_context_lens_tensor[1],
            max_decode_query_len = max_decode_query_len_right,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            tree_attention_masks_tensor = None,
            block_tables_list = None,
            encoder_seq_lens = encoder_seq_lens_right,
            encoder_seq_lens_tensor = encoder_seq_lens_tensor_right,
            max_encoder_seq_len = max_encoder_seq_len_right,
            num_encoder_tokens = num_encoder_tokens_right,
            cross_slot_mapping = cross_slot_mapping_right,
            cross_block_tables = cross_block_tables_right,
        )
        
    if isinstance(model_input.attn_metadata, FlashMLAMetadata):
        attn_metadata_left = FlashMLAMetadata(
            num_prefills = num_prefills_left,
            num_prefill_tokens = num_prefill_tokens_left,
            num_decode_tokens = num_decode_tokens_left,
            slot_mapping = split_slot_mapping[0],
            multi_modal_placeholder_index_maps = model_input.attn_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            input_positions = split_input_positions[0],
            seq_lens = seq_lens_left,
            seq_lens_tensor = split_seq_lens_tensor[0],
            max_prefill_seq_len = max_prefill_seq_len_left,
            max_decode_seq_len = max_decode_seq_len_left,
            context_lens_tensor = split_context_lens_tensor[0],
            block_tables = split_block_tables[0],
            max_query_len = max_query_len_left,
            max_decode_query_len = max_decode_query_len_left,
            query_start_loc = query_start_loc_left,
            seq_start_loc = seq_start_loc_left,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            head_dim = model_input.attn_metadata.head_dim,
            is_profile_run = model_input.attn_metadata.is_profile_run,
            context_chunk_cu_seq_lens=model_input.attn_metadata.context_chunk_cu_seq_lens, 
            context_chunk_starts=model_input.attn_metadata.context_chunk_starts, 
            context_chunk_seq_tot=model_input.attn_metadata.context_chunk_seq_tot, 
            context_chunk_max_seq_lens=model_input.attn_metadata.context_chunk_max_seq_lens, 
            context_chunk_workspace=model_input.attn_metadata.context_chunk_workspace, 
            decode_tile_scheduler_metadata=model_input.attn_metadata.decode_tile_scheduler_metadata, 
            decode_num_splits=model_input.attn_metadata.decode_num_splits
        )
        attn_metadata_right = FlashMLAMetadata(
            num_prefills = num_prefills_right,
            num_prefill_tokens = num_prefill_tokens_right,
            num_decode_tokens = num_decode_tokens_right,
            slot_mapping = split_slot_mapping[1],
            multi_modal_placeholder_index_maps = model_input.attn_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            input_positions = split_input_positions[1],
            seq_lens = seq_lens_right,
            seq_lens_tensor = split_seq_lens_tensor[1],
            max_prefill_seq_len = max_prefill_seq_len_right,
            max_decode_seq_len = max_decode_seq_len_right,
            context_lens_tensor = split_context_lens_tensor[1],
            block_tables = split_block_tables[1],
            max_query_len = max_query_len_right,
            max_decode_query_len = max_decode_query_len_right,
            query_start_loc = query_start_loc_right,
            seq_start_loc = seq_start_loc_right,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            head_dim = model_input.attn_metadata.head_dim,
            is_profile_run = model_input.attn_metadata.is_profile_run,
            context_chunk_cu_seq_lens=model_input.attn_metadata.context_chunk_cu_seq_lens, 
            context_chunk_starts=model_input.attn_metadata.context_chunk_starts, 
            context_chunk_seq_tot=model_input.attn_metadata.context_chunk_seq_tot, 
            context_chunk_max_seq_lens=model_input.attn_metadata.context_chunk_max_seq_lens, 
            context_chunk_workspace=model_input.attn_metadata.context_chunk_workspace, 
            decode_tile_scheduler_metadata=model_input.attn_metadata.decode_tile_scheduler_metadata, 
            decode_num_splits=model_input.attn_metadata.decode_num_splits
        )
    if isinstance(model_input.attn_metadata, FlashAttentionMetadata):
        attn_metadata_left = FlashAttentionMetadata(
            seq_lens_tensor = split_seq_lens_tensor[0],
            max_decode_seq_len = max_decode_seq_len_left,
            block_tables = split_block_tables[0],
            num_prefills = num_prefills_left,
            num_prefill_tokens = num_prefill_tokens_left,
            num_decode_tokens = num_decode_tokens_left,
            slot_mapping = split_slot_mapping[0],
            multi_modal_placeholder_index_maps = {},
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            seq_lens = seq_lens_left,
            max_prefill_seq_len = max_prefill_seq_len_left,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            max_query_len = max_query_len_left,
            query_start_loc = query_start_loc_left,
            seq_start_loc = seq_start_loc_left,
            context_lens_tensor = split_context_lens_tensor[0],
            max_decode_query_len = max_decode_query_len_left,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            encoder_seq_lens_tensor = encoder_seq_lens_tensor_left,
            max_encoder_seq_len = max_encoder_seq_len_left,
            num_encoder_tokens = num_encoder_tokens_left,
        )
        attn_metadata_right = FlashAttentionMetadata(
            seq_lens_tensor = split_seq_lens_tensor[1],
            max_decode_seq_len = max_decode_seq_len_right,
            block_tables = split_block_tables[1],
            num_prefills = num_prefills_right,
            num_prefill_tokens = num_prefill_tokens_right,
            num_decode_tokens = num_decode_tokens_right,
            slot_mapping = split_slot_mapping[1],
            multi_modal_placeholder_index_maps = {},
            enable_kv_scales_calculation = model_input.attn_metadata.enable_kv_scales_calculation,
            seq_lens = seq_lens_right,
            max_prefill_seq_len = max_prefill_seq_len_right,
            use_cuda_graph = model_input.attn_metadata.use_cuda_graph,
            max_query_len = max_query_len_right,
            query_start_loc = query_start_loc_right,
            seq_start_loc = seq_start_loc_right,
            context_lens_tensor = split_context_lens_tensor[1],
            max_decode_query_len = max_decode_query_len_right,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            encoder_seq_lens_tensor = encoder_seq_lens_tensor_right,
            max_encoder_seq_len = max_encoder_seq_len_right,
            num_encoder_tokens = num_encoder_tokens_right,
        )         
    model_input_left = ModelInputForGPUWithSamplingMetadata(
        input_tokens=split_input_tokens[0],
        input_positions=split_input_positions[0],
        token_types=None,
        seq_lens=seq_lens_left,
        query_lens=query_lens_left,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=attn_metadata_left,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=request_ids_to_seq_ids_left,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        async_callback=model_input.async_callback,
        scheduler_outputs=model_input.scheduler_outputs,
        previous_hidden_states=previous_hidden_states_left,
        sampling_metadata=None, #TBO does not require sampling_stetadata
        is_prompt=model_input.is_prompt,
    )
    model_input_right = ModelInputForGPUWithSamplingMetadata(
        input_tokens=split_input_tokens[1],
        input_positions=split_input_positions[1],
        token_types=None,
        seq_lens=seq_lens_right,
        query_lens=query_lens_right,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=attn_metadata_right,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=request_ids_to_seq_ids_right,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        async_callback=model_input.async_callback,
        scheduler_outputs=model_input.scheduler_outputs,
        previous_hidden_states=previous_hidden_states_right,
        sampling_metadata=None, #TBO does not require sampling_stetadata
        is_prompt=model_input.is_prompt,
    )
    return model_input_left, model_input_right

def split_capture_attention_metadata(attn_metadata, batch_size_left, batch_size_right):
    batch_size_split = [batch_size_left, batch_size_right]
    split_seq_lens_tensor = torch.split(attn_metadata.seq_lens_tensor, batch_size_split, dim=0)
    split_block_tables = torch.split(attn_metadata.block_tables, batch_size_split, dim=0)
    split_slot_mapping = torch.split(attn_metadata.slot_mapping, batch_size_split, dim=0)
    if isinstance(attn_metadata, ROCmFlashAttentionMetadata):
        attn_metadata_left = ROCmFlashAttentionMetadata(
            seq_lens_tensor = split_seq_lens_tensor[0],
            max_decode_seq_len = attn_metadata.max_decode_seq_len,
            block_tables = split_block_tables[0],
            num_prefills = 0,
            num_prefill_tokens = 0,
            num_decode_tokens = batch_size_left,
            slot_mapping = split_slot_mapping[0],
            multi_modal_placeholder_index_maps = attn_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation = attn_metadata.enable_kv_scales_calculation,
            seq_lens = None,
            max_prefill_seq_len = 0,
            use_cuda_graph = attn_metadata.use_cuda_graph,
            max_query_len = 1,
            query_start_loc = None,
            seq_start_loc = None,
            context_lens_tensor = None,
            max_decode_query_len = 1,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            tree_attention_masks_tensor = None,
            block_tables_list = None,
            encoder_seq_lens = None,
            encoder_seq_lens_tensor = None,
            max_encoder_seq_len = None,
            num_encoder_tokens = None,
            cross_slot_mapping = None,
            cross_block_tables = None,
        )
        attn_metadata_right = ROCmFlashAttentionMetadata(
            seq_lens_tensor = split_seq_lens_tensor[1],
            max_decode_seq_len = attn_metadata.max_decode_seq_len,
            block_tables = split_block_tables[1],
            num_prefills = 0,
            num_prefill_tokens = 0,
            num_decode_tokens = batch_size_right,
            slot_mapping = split_slot_mapping[1],
            multi_modal_placeholder_index_maps = attn_metadata.multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation = attn_metadata.enable_kv_scales_calculation,
            seq_lens = None,
            max_prefill_seq_len = 0,
            use_cuda_graph = attn_metadata.use_cuda_graph,
            max_query_len = 1,
            query_start_loc = None,
            seq_start_loc = None,
            context_lens_tensor = None,
            max_decode_query_len = 1,
            _cached_prefill_metadata = None,
            _cached_decode_metadata = None,
            tree_attention_masks_tensor = None,
            block_tables_list = None,
            encoder_seq_lens = None,
            encoder_seq_lens_tensor = None,
            max_encoder_seq_len = None,
            num_encoder_tokens = None,
            cross_slot_mapping = None,
            cross_block_tables = None,
        )
    else:
        print("tbo:not surpport in cuda-graph ", type(attn_metadata))
    return attn_metadata_left, attn_metadata_right
