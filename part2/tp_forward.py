import torch
import torch.distributed as dist
class column_parallel_linear_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tp_group):
        ctx.save_for_backward(x, weight, bias)
        ctx.tp_group = tp_group
        x = torch.matmul(x, weight.T) + bias 
        return x
    
    def backward(ctx, grad_output):
        ### TODO: Implement tensor parallel linear backward
        raise NotImplementedError
        ### TODO END
        
        return grad_input, grad_weight, grad_bias , None
    

class row_parallel_linear_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, tp_group):
        ctx.save_for_backward(x, weight, bias)
        ctx.tp_group = tp_group
        x = torch.matmul(x, weight.T) + bias
        return x
    def backward(ctx, grad_output):
        ### TODO: Implement tensor parallel linear backward
        raise NotImplementedError
        ### TODO END
        
        return grad_input, grad_weight, grad_bias, None
    
class row_parallel_embedding_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.save_for_backward(x)
        ctx.tp_group = tp_group
        ### TODO: Implement tensor parallel embedding forward
        raise NotImplementedError
        ### TODO END
        
        return x
    
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        ### TODO: Implement tensor parallel embedding backward
        raise NotImplementedError
        ### TODO END
        
        return grad_output, None, None
    

class parallel_attention_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tp_group):
        ctx.save_for_backward(x)
        ctx.tp_group = tp_group
        ### TODO: Implement tensor parallel attention forward
        raise NotImplementedError
        ### TODO END
        
        return x
    
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        ### TODO: Implement tensor parallel attention backward
        raise NotImplementedError
        ### TODO END
        
        return grad_output, None
