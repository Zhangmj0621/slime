from dataclasses import dataclass, field
from typing import Dict

from slime.backends.megatron_utils.actor import MegatronTrainRayActor
from slime.ray.rollout import RolloutManager

@dataclass
class ParamMeta:
    original_name: str
    param_size: int 
    
def set_routing_table(
    actor_param_metadata: list[Dict[str, ParamMeta]],
    rollout_param_metadata: list[list[list[str]]],
):
    """Set the routing table for parameter updates.

    Args:
        actor_param_metadata (list[Dict[str, ParamMeta]]): Parameter metadata for actor workers.
        rollout_param_metadata (list[list[str]]): Parameter metadata for rollout workers.
    """
    
    actor_worker_num = len(actor_param_metadata)
    rollout_engine_num = len(rollout_param_metadata)
    rollout_worker_num_per_engine = len(rollout_param_metadata[0])

    routing_table = []

    # sendbytes of each actor worker
    sendbytes_per_actor = [0 for _ in range(actor_worker_num)]

    for engine_index in range(0, rollout_engine_num):
        engine_routing_table = []
        for rollout_worker_index in range(0, rollout_worker_num_per_engine):
            rollout_worker_param = rollout_param_metadata[engine_index][rollout_worker_index]
            worker_routing_table = {}

            # find the corresponding actor worker for each rollout worker parameter
            for param_name in rollout_worker_param:
                corresponding_actor_worker_list = []
                for actor_worker_index in range(0, actor_worker_num):
                    actor_param_meta_dict = actor_param_metadata[actor_worker_index]
                    if param_name in actor_param_meta_dict:
                        corresponding_actor_worker_list.append(actor_worker_index)
                    
                # choose the actor worker with the least sendbytes
                chosen_actor_worker = min(
                    corresponding_actor_worker_list, key=lambda idx: sendbytes_per_actor[idx]
                )
                param_size = actor_param_metadata[chosen_actor_worker][param_name].param_size
                sendbytes_per_actor[chosen_actor_worker] += param_size
                worker_routing_table[param_name] = [chosen_actor_worker, actor_param_metadata[chosen_actor_worker][param_name].original_name]
            engine_routing_table.append(worker_routing_table)
        routing_table.append(engine_routing_table)
    return routing_table
            

        
    
