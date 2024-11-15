import torch
import torch.distributed as dist

def align_size(data, bucket_size):
    shape = data.shape
    data = data.view(-1)
    if data.numel() % bucket_size != 0:
        data = torch.cat([data, torch.zeros(bucket_size - data.numel() % bucket_size, dtype=data.dtype, device=data.device)])
    return data

class SimpleOptimizer:
    def __init__(self, model, 
                 lr=0.001, 
                 weight_decay=0.0,
                 gradient_accumulation_steps=1):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.total_steps = 0

    def step(self):

        self.total_steps += 1
        if self.total_steps % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

class ZeroOptimizer:
    def __init__(self, model, 
                 dp_group=None, 
                 lr=0.001, 
                 weight_decay=0.0,
                 gradient_accumulation_steps=1,
                 stage=0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.stage = stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if dp_group.size() == 1:
            self.dp_group = dist.group.WORLD
        else:
            self.dp_group = dp_group

        self.dp_size = self.dp_group.size()
        self.dp_rank = self.dp_group.rank()
        
        if self.stage == 0:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.stage == 1:
            self.pbuckets = []
            self.partition_parameters()
            self.optimizer = torch.optim.AdamW(self.pbuckets, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"We only support stage 0 and 1, but got {self.stage}") 
        self.parameter_size = sum(1 for _ in self.model.parameters())
        self.total_steps = 0

    def partition_parameters(self):
        
        ### TODO: Partition parameters equally among processes
        ### Hint: You should forloop through all parameters, and maintain a param shard of each parameter, 
        ### and parameterize the param shard so you can send them into optimizer
        raise NotImplementedError()
        for param in self.model.parameters():
            param_size = param.size()
            shard_size = param_size[0] // self.dp_size
            pbucket = []
            for i in range(self.dp_size):
                start = i * shard_size
                end = (i + 1) * shard_size
                pbucket.append(param.data[start:end].clone())
            self.pbuckets.append(pbucket)
            print(self.pbuckets)
        ### TODO END

    def step(self):

        self.total_steps += 1

        if self.total_steps % self.gradient_accumulation_steps == 0:
            if self.stage == 0: #self.stage means the optimization stage
                ### TODO: all reduce the grad, and then step
                print('start stage 0 reduce')
                #print(self.gradient_accumulation_steps)
                for param in self.model.parameters():
                    #world_size = dist.get_world_size(group=self.dp_group)
                    #print(world_size)
                    param.grad /= (self.gradient_accumulation_steps * self.parameter_size)
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                    
                self.optimizer.step()
                ### TODO END
            elif self.stage == 1:
                ### TODO: Forloop through gradients of all parameters, and do reduce scatter, 
                ### and send the grad into self.pbuckets
                raise NotImplementedError()
                print('start stage 1 reduce')
                for i, param in enumerate(self.model.parameters()):
                    dist.reduce_scatter(param.grad, self.pbuckets[i], op=dist.ReduceOp.SUM)
                ### TODO END

                self.optimizer.step()
                
                ### TODO: Forloop through all param shards, and do all gather to get a global model param
                ### and update the param of the model
                for i, param in enumerate(self.model.parameters()):
                    dist.all_gather(param.data, self.pbuckets[i])
                
                ### TODO END

                self.optimizer.zero_grad()
                for param in self.model.parameters():
                    param.grad = None