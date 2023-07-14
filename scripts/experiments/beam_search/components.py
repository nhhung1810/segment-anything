import torch
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from uuid import UUID


@dataclass
class BeamSearchOptionData:
    obj_id: UUID = field(default_factory=uuid.uuid4)
    prev_id: Optional[UUID] = None
    sigmoid_threshold: float = 0.0
    score: float = 0.0
    prev_score: float = 0.0
    frame_idx: Optional[int] = None
    next_frame_idx: Optional[int] = None
    mask_logits: torch.Tensor = None
    cyclic_count: int = 0

    def get_mask(self) -> torch.Tensor:
        if self.prev_id is None:
            return self.mask_logits

        return torch.sigmoid(self.mask_logits) > self.sigmoid_threshold

    def get_confidence_score(
        self, scoring_fn: Callable[[torch.Tensor, float], float]
    ) -> float:
        return scoring_fn(self.mask_logits, self.sigmoid_threshold)


@dataclass
class BeamSearchTracing:
    data: BeamSearchOptionData = None
    children: List[BeamSearchOptionData] = field(default_factory=list)


class Tracing:
    def __init__(self) -> None:
        self.trace_dict: Dict[str, BeamSearchTracing] = {}
        pass

    def add_multi(self, objs: List[BeamSearchOptionData]):
        for obj in objs:
            self.add(obj)
            pass

    def add(self, obj: BeamSearchOptionData):
        # Register itself into the system
        self.trace_dict[obj.obj_id] = BeamSearchTracing(data=obj)
        # Link the current object
        prev_data = self.trace_dict.get(obj.prev_id, None)
        if prev_data is None:
            return
        prev_data.children.append(obj)
        pass

    def flatten(self):
        result = []
        for value in self.trace_dict.values():
            result.extend(value.children)
        return result

    def tracing(self, prev_id) -> List[BeamSearchOptionData]:
        # Safe-guard the tracing
        counter = 1000
        prev = self.trace_dict[prev_id]
        result = []
        while prev.data.prev_id is not None and counter > 0:
            result.append(prev.data)
            prev = self.trace_dict[prev.data.prev_id]
            counter = counter - 1
            pass
        return result
