
import gc
import os
import queue
import threading
from typing import List, Optional, Tuple
import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.multimodal.inputs import MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.two_batch_overlap.forward_context import init_tbo_forward_context
from vllm.two_batch_overlap.model_input_split import is_supported_attention_metadata, split_capture_attention_metadata, split_model_input
from vllm.logger import init_logger
from vllm.profiler.prof import profile
from vllm import envs
from vllm.utils import weak_ref_tensor

tbo_one_stream = os.environ.get('VLLM_TBO_ONE_STREAM') == '1'

logger = init_logger(__name__)

tbo_step_stream = None
all_reduce_stream = None

class TwoBatchOverlap():
    def __init__(self):
        global tbo_step_stream
        global all_reduce_stream
        self.model_input_left_queue = queue.Queue()
        self.model_input_right_queue = queue.Queue()
        self.states_left_queue = queue.Queue()
        self.states_right_queue = queue.Queue()
        self.left_thread = None
        self.right_thread = None
        self.left_tid = 0
        self.right_tid = 0
        self.sem_left = threading.Semaphore(0)
        self.sem_right = threading.Semaphore(0)
        self.left_first = False
        self.tbo_running = False
        self.tbo_in_capture = False
        if tbo_step_stream == None:
            tbo_step_stream = torch.cuda.Stream()
            all_reduce_stream = torch.cuda.Stream()
        self.step_event = torch.cuda.Event(enable_timing=False)
        self.event_left_c2t = torch.cuda.Event(enable_timing=False)
        self.event_right_c2t = torch.cuda.Event(enable_timing=False)
        self.event_left_t2c = torch.cuda.Event(enable_timing=False)
        self.event_right_t2c = torch.cuda.Event(enable_timing=False)

    def init_tbo_thread(self):
        self.model_input_left_queue.empty()
        self.model_input_right_queue.empty()
        self.left_thread = threading.Thread(target=self.thread_two_batch_overlap, args=(self.model_input_left_queue,))
        self.left_thread.start()
        self.right_thread = threading.Thread(target=self.thread_two_batch_overlap, args=(self.model_input_right_queue,))
        self.right_thread.start()
        logger.info('tbo:two batch overlap start')

    def finish_thread(self):
        self.left_thread.join()
        self.left_thread = None
        self.right_thread.join()
        self.right_thread = None
        
    @torch.inference_mode()
    def thread_two_batch_overlap(self, queue):
        is_left_thread = False
        tid = threading.get_ident()
        if queue == self.model_input_left_queue:
            self.left_tid = tid
            is_left_thread = True
            init_tbo_forward_context(True, self.left_tid)
        else:
            self.right_tid = tid
            init_tbo_forward_context(False, self.right_tid)
        with torch.cuda.stream(tbo_step_stream):
            model_input = queue.get()
            profile.ProfRangePush('start')
            self.tbo_thread_synchronize(tid)
            model_kwargs = None
            intermediate_tensors = None
            if is_left_thread:
                model_kwargs = self.model_kwargs_left
                intermediate_tensors = self.intermediate_tensors_left
            else:
                model_kwargs = self.model_kwargs_right
                intermediate_tensors = self.intermediate_tensors_right
            hidden_or_intermediate_states = None
            if self.tbo_in_capture:
                if is_left_thread:
                    attn_metadata = self.attn_metadata_left
                    input_tokens = self.input_tokens_left
                    input_positions = self.split_input_positions[0]
                else:
                    attn_metadata = self.attn_metadata_right
                    input_tokens = self.input_tokens_right
                    input_positions = self.split_input_positions[1]
                with set_forward_context(attn_metadata,
                                        self.vllm_config, self.virtual_engine):
                    hidden_or_intermediate_states = self.model_executable(
                        input_ids=input_tokens,
                        positions=input_positions,
                        intermediate_tensors=intermediate_tensors,
                        **MultiModalKwargs.as_kwargs(self.multi_modal_kwargs,
                                                        device=self.self_device),
                        **model_kwargs,
                    )
            elif model_input != None:
                with set_forward_context(model_input.attn_metadata,
                                        self.vllm_config, self.virtual_engine):
                    hidden_or_intermediate_states = self.model_executable(
                        input_ids=model_input.input_tokens,
                        positions=model_input.input_positions,
                        intermediate_tensors=intermediate_tensors,
                        **MultiModalKwargs.as_kwargs(self.multi_modal_kwargs,
                                                        device=self.self_device),
                        **self.seqlen_agnostic_kwargs,
                        **model_kwargs,
                    )
            if is_left_thread:
                self.sem_right.release()
                self.states_left_queue.put(hidden_or_intermediate_states)
            else:
                self.states_right_queue.put(hidden_or_intermediate_states)
            profile.ProfRangePop()

    def tbo_thread_synchronize(self, tid):
        if tid == self.left_tid:
            if not self.left_first:
                self.sem_right.release()
            self.left_first = False
            profile.ProfRangePop()
            self.sem_left.acquire()
            profile.ProfRangePush('left')
            return self.event_left_c2t, self.event_left_t2c
        else:
            self.sem_left.release()
            profile.ProfRangePop()
            self.sem_right.acquire()
            profile.ProfRangePush('right')
            return self.event_right_c2t, self.event_right_t2c

    def set_model_input(self,
                        model_input_left, 
                        model_input_right, 
                        vllm_config,
                        virtual_engine,
                        model_executable,
                        intermediate_tensors_left,
                        intermediate_tensors_right,
                        multi_modal_kwargs,
                        self_device,
                        seqlen_agnostic_kwargs,
                        model_kwargs_left,
                        model_kwargs_right):
        self.vllm_config = vllm_config
        self.virtual_engine = virtual_engine
        self.model_executable = model_executable
        self.intermediate_tensors_left = intermediate_tensors_left
        self.intermediate_tensors_right = intermediate_tensors_right
        self.multi_modal_kwargs = multi_modal_kwargs
        self.self_device = self_device
        self.seqlen_agnostic_kwargs = seqlen_agnostic_kwargs
        self.model_kwargs_left = model_kwargs_left
        self.model_kwargs_right = model_kwargs_right
        self.model_input_left_queue.put(model_input_left)
        self.model_input_right_queue.put(model_input_right)

    def set_capture_model_input(self,
                                input_tokens_left, 
                                input_tokens_right, 
                                split_input_positions, 
                                vllm_config,
                                virtual_engine,
                                runner_model,
                                runner_device,
                                intermediate_tensors_left,
                                intermediate_tensors_right,
                                model_kwargs_left,
                                model_kwargs_right,
                                attn_metadata_left, 
                                attn_metadata_right):
        self.input_tokens_left  = input_tokens_left
        self.input_tokens_right  = input_tokens_right
        self.split_input_positions = split_input_positions
        self.vllm_config = vllm_config
        self.virtual_engine = virtual_engine
        self.model_executable = runner_model
        self.self_device = runner_device
        self.intermediate_tensors_left = intermediate_tensors_left
        self.intermediate_tensors_right = intermediate_tensors_right
        self.model_kwargs_left = model_kwargs_left
        self.model_kwargs_right = model_kwargs_right
        self.attn_metadata_left = attn_metadata_left
        self.attn_metadata_right = attn_metadata_right
        self.model_input_left_queue.put(None)
        self.model_input_right_queue.put(None)

    def get_model_output(self):
        states_left = self.states_left_queue.get()
        states_right = self.states_right_queue.get()
        return states_left, states_right
    
tbo_obj = None

def init_two_batch_overlap():
    global tbo_obj
    if tbo_obj == None:
        tbo_obj = TwoBatchOverlap()
    tbo_obj.init_tbo_thread()

def tbo_all_reduce(obj):
    if envs.VLLM_ENABLE_TBO and tbo_obj != None and tbo_obj.tbo_running:
        tid = threading.get_ident()
        if not tbo_one_stream:
            if tid == tbo_obj.left_tid:
                event_c2t, event_t2c = tbo_obj.event_left_c2t, tbo_obj.event_left_t2c
            else:
                event_c2t, event_t2c = tbo_obj.event_right_c2t, tbo_obj.event_right_t2c
            event_c2t.record()
            with torch.cuda.stream(all_reduce_stream):
                all_reduce_stream.wait_event(event_c2t)
                output = tensor_model_parallel_all_reduce(obj)
                event_t2c.record()
            tbo_obj.tbo_thread_synchronize(tid)
            tbo_step_stream.wait_event(event_t2c)
        else:
            output = tensor_model_parallel_all_reduce(obj)
            tbo_obj.tbo_thread_synchronize(tid)
        return output
    return tensor_model_parallel_all_reduce(obj) 

def merge_model_output(states_left, states_right):
    if isinstance(states_left, IntermediateTensors):
        output_map = {}
        for key in states_left.tensors:
            output_map[key] = torch.concat([states_left.tensors[key], states_right.tensors[key]], dim=0)
        output = IntermediateTensors(output_map)
    else:
        output = torch.concat([states_left, states_right], dim=0)
    return output

def tbo_model_executable(
        model_input, 
        vllm_config,
        virtual_engine,
        model_executable,
        intermediate_tensors,
        multi_modal_kwargs,
        self_device,
        seqlen_agnostic_kwargs,
        model_kwargs,
    ):
    is_support = is_supported_attention_metadata(model_input.attn_metadata)
    if not is_support:
        logger.info("tbo:not surpport yet ", type(model_input.attn_metadata))
    batch_size = len(model_input.attn_metadata.seq_lens)
    is_decode_tbo_invalid = not model_input.is_prompt and (
        envs.VLLM_TBO_DECODE_BS < 2 or 
        batch_size < envs.VLLM_TBO_DECODE_BS or 
        model_input.attn_metadata.use_cuda_graph)
    if batch_size == 1 or \
        is_decode_tbo_invalid or \
        not is_support:
        with set_forward_context(model_input.attn_metadata,
                                    vllm_config, virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(multi_modal_kwargs,
                                                device=self_device),
                **seqlen_agnostic_kwargs,
                **model_kwargs,
            )
        return hidden_or_intermediate_states
    profile.ProfRangePush('tbo_model_executable')
    init_two_batch_overlap()
    tbo_obj.tbo_running = True
    tbo_obj.left_first = True
    batch_size_left = int(batch_size / 2)
    batch_size_right = batch_size_left
    if batch_size % 2 == 1:
        batch_size_right += 1
    
    model_input_left, model_input_right = split_model_input(model_input, self_device, batch_size_left, batch_size_right)
    
    model_kwargs_left = model_kwargs.copy()
    model_kwargs_right = model_kwargs.copy()
    intermediate_tensors_left = None
    intermediate_tensors_right = None
    if "previous_hidden_states" in model_kwargs:
        previous_hidden_states = model_kwargs["previous_hidden_states"]
        query_tokens_split = [sum(model_input.query_lens[0:batch_size_left]), sum(model_input.query_lens[batch_size_left:])]
        split_previous_hidden_states = torch.split(previous_hidden_states, query_tokens_split, dim=0)
        model_kwargs_left["previous_hidden_states"] = split_previous_hidden_states[0]
        model_kwargs_right["previous_hidden_states"] = split_previous_hidden_states[1]
    if intermediate_tensors != None:
        query_tokens_split = [sum(model_input.query_lens[0:batch_size_left]), sum(model_input.query_lens[batch_size_left:])]
        intermediate_tensors_left = {}
        intermediate_tensors_right = {}
        for key in intermediate_tensors.tensors:
            split_intermediate_tensors = torch.split(intermediate_tensors.tensors[key], query_tokens_split, dim=0)
            intermediate_tensors_left[key] = split_intermediate_tensors[0]
            intermediate_tensors_right[key] = split_intermediate_tensors[1]
        intermediate_tensors_left = IntermediateTensors(intermediate_tensors_left)
        intermediate_tensors_right = IntermediateTensors(intermediate_tensors_right)

    tbo_obj.step_event.record()
    current_stream = torch.cuda.current_stream()
    with torch.cuda.stream(tbo_step_stream):
        tbo_step_stream.wait_event(tbo_obj.step_event)
        tbo_obj.set_model_input(model_input_left, 
                                model_input_right, 
                                vllm_config,
                                virtual_engine,
                                model_executable,
                                intermediate_tensors_left,
                                intermediate_tensors_right,
                                multi_modal_kwargs,
                                self_device,
                                seqlen_agnostic_kwargs,
                                model_kwargs_left,
                                model_kwargs_right)
        
        states_left, states_right = tbo_obj.get_model_output()

        hidden_or_intermediate_states = merge_model_output(states_left, states_right)
        tbo_obj.tbo_running = False
        tbo_obj.step_event.record()
        tbo_obj.finish_thread()
    current_stream.wait_event(tbo_obj.step_event)
    profile.ProfRangePop()
    return hidden_or_intermediate_states

def _run_once(vllm_config, virtual_engine,
        runner,
        self_device,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_inputs: Optional[IntermediateTensors],
        attn_metadata: AttentionMetadata,
        stream: torch.cuda.Stream,
        **kwargs):
    global tbo_step_stream
    stream_back = tbo_step_stream
    tbo_step_stream = stream
    init_two_batch_overlap()
    tbo_obj.left_first = True
    decode_batch_size = input_ids.shape[0]
    batch_size_left = int(decode_batch_size / 2)
    batch_size_right = decode_batch_size - batch_size_left
    query_tokens_split = [batch_size_left, batch_size_right]
    input_tokens_left, input_tokens_right = torch.split(input_ids, query_tokens_split, dim=0)
    split_input_positions = torch.split(positions, query_tokens_split, dim=0)
    model_kwargs_left = kwargs.copy()
    model_kwargs_right = kwargs.copy()
    intermediate_tensors_left = None
    intermediate_tensors_right = None
    if "previous_hidden_states" in kwargs:
        previous_hidden_states = kwargs["previous_hidden_states"]
        split_previous_hidden_states = torch.split(previous_hidden_states, query_tokens_split, dim=0)
        model_kwargs_left["previous_hidden_states"] = split_previous_hidden_states[0]
        model_kwargs_right["previous_hidden_states"] = split_previous_hidden_states[1]
    if intermediate_inputs != None:
        query_tokens_split = [batch_size_left, batch_size_right]
        intermediate_tensors_left = {}
        intermediate_tensors_right = {}
        for key in intermediate_inputs.tensors:
            split_intermediate_tensors = torch.split(intermediate_inputs.tensors[key], query_tokens_split, dim=0)
            intermediate_tensors_left[key] = split_intermediate_tensors[0]
            intermediate_tensors_right[key] = split_intermediate_tensors[1]
        intermediate_tensors_left = IntermediateTensors(intermediate_tensors_left)
        intermediate_tensors_right = IntermediateTensors(intermediate_tensors_right)
    attn_metadata_left, attn_metadata_right = split_capture_attention_metadata(attn_metadata, batch_size_left, batch_size_right)
    tbo_obj.tbo_running = True
    tbo_obj.tbo_in_capture = True
    tbo_obj.set_capture_model_input(input_tokens_left, 
                                    input_tokens_right, 
                                    split_input_positions, 
                                    vllm_config,
                                    virtual_engine,
                                    runner.model,
                                    self_device,
                                    intermediate_tensors_left,
                                    intermediate_tensors_right,
                                    model_kwargs_left,
                                    model_kwargs_right, 
                                    attn_metadata_left, 
                                    attn_metadata_right)

    states_left, states_right = tbo_obj.get_model_output()
    output_hidden_or_intermediate_states = merge_model_output(states_left, states_right)
    tbo_obj.tbo_in_capture = False
    tbo_obj.tbo_running = False
    tbo_obj.finish_thread()
    tbo_step_stream = stream_back
    return output_hidden_or_intermediate_states

def tbo_capture(vllm_config, virtual_engine, _NUM_WARMUP_ITERS, 
        runner,
        self_device,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_inputs: Optional[IntermediateTensors],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs):
    for i in range(_NUM_WARMUP_ITERS):
        _run_once(vllm_config, 
                    virtual_engine,
                    runner,
                    self_device,
                    input_ids,
                    positions,
                    intermediate_inputs,
                    attn_metadata,
                    torch.cuda.current_stream(),
                    **kwargs)
        torch.cuda.synchronize()
    runner._graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(runner._graph, pool=memory_pool, stream=stream):
        output_hidden_or_intermediate_states = _run_once(vllm_config, 
                                                            virtual_engine, 
                                                            runner,
                                                            self_device,
                                                            input_ids,
                                                            positions,
                                                            intermediate_inputs,
                                                            attn_metadata,
                                                            torch.cuda.current_stream(),
                                                            **kwargs)
        if isinstance(output_hidden_or_intermediate_states, torch.Tensor):
            hidden_or_intermediate_states = weak_ref_tensor(
                output_hidden_or_intermediate_states)
        elif isinstance(output_hidden_or_intermediate_states,
                        IntermediateTensors):
            hidden_or_intermediate_states = IntermediateTensors(
                tensors={
                    key: weak_ref_tensor(value)
                    for key, value in
                    output_hidden_or_intermediate_states.tensors.items()
                })

        del output_hidden_or_intermediate_states
        # make sure `output_hidden_or_intermediate_states` is deleted
        # in the graph's memory pool
        gc.collect()
    torch.cuda.synchronize()
     
    # Save the input and output buffers.
    runner.input_buffers = {
        "input_ids":
        input_ids,
        "positions":
        positions,
        "kv_caches":
        kv_caches,
        **runner.attn_state.get_graph_input_buffers(
            attn_metadata, runner._is_encoder_decoder_model),
        **kwargs,
    }
    if intermediate_inputs is not None:
        runner.input_buffers.update(intermediate_inputs.tensors)
    if get_pp_group().is_last_rank:
        runner.output_buffers = {
            "hidden_states": hidden_or_intermediate_states
        }
    else:
        runner.output_buffers = hidden_or_intermediate_states
