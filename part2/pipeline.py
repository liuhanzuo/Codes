import torch.nn as nn
import torch
import torch.distributed as dist

class NoPipe:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
    
    def forward_backward(self, input_ids, labels):
        outputs = self.model(input_ids)
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        return loss

class SimplePipe:
    def __init__(self, model, criterion , rank, pp_group, tp_size):
        super(SimplePipe, self).__init__()
        self.model = model
        self.criterion = criterion
        self.rank = rank
        self.pp_group = pp_group
        self.tp_size = tp_size
        self.next_rank = (self.rank + tp_size)
        self.prev_rank = (self.rank - tp_size)
        print(f'[rank {self.rank}] PP GROUP {self.pp_group.rank()}')
        self.is_first_pp = self.pp_group.rank() == 0
        self.is_last_pp = self.pp_group.rank() == self.pp_group.size() - 1

    # def forward(self, x):
    #     return self.model(x)
    
    def forward_imp(self, input_ids, labels):   
        ### TODO: Implement pipeline forward
        raise NotImplementedError
        ### TODO END
        return inputs, outputs , loss

    def backward_imp(self, inputs, outputs, loss):
        ### TODO: Implement pipeline backward
        raise NotImplementedError
        ### TODO END
        return loss


    def forward_backward(self, input_ids, labels):
        inputs, outputs, loss = self.forward_imp(input_ids, labels)
        loss = self.backward_imp(inputs, outputs, loss)
        return loss       

class GPipe:
    def __init__(self, model, criterion, rank, pp_group, tp_size):
        super(GPipe, self).__init__()
        self.model = model
        self.criterion = criterion
        self.rank = rank
        self.pp_group = pp_group
        self.tp_size = tp_size
        self.next_rank = (self.rank + tp_size)
        self.prev_rank = (self.rank - tp_size)
        self.is_first_pp = self.pp_group.rank() == 0
        self.is_last_pp = self.pp_group.rank() == self.pp_group.size() - 1

    def forward_imp(self, input_ids, labels):   
        ### TODO: Implement pipeline forward (can be same as SimplePipe)
        raise NotImplementedError
        ### TODO END
        return inputs, outputs , loss
    
    def backward_imp(self, inputs, outputs, loss):
        ### TODO: Implement pipeline backward (can be same as SimplePipe)
        raise NotImplementedError
        ### TODO END
        return loss

    def forward_backward(self, samples):
        ### TODO: Implement pipeline forward and backward
        raise NotImplementedError
        ### TODO END
        return losses
