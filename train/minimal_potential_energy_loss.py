import torch
import torch.distributed as dist
class MPELoss_with_backward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, H, voxel, Ke_hard, Fe_hard,Ke_soft,Fe_soft):
        """A function to compute minimal poyential energy loss

        Args:
            ctx (object): Autograd function context
            input (Tensor): Batched displacement output by network, it should be b*18*N*N*N torch.cuda.floatTensor
            H (object): Homo3D instance
            voxel (Tensor): Batched voxel, it should be b*N*N*N torch.cuda.floatTensor
            Ke (Tensor): Batched elelment stiffness matrix, it should be b*24*24 torch.cuda.floatTensor
            Fe (Tensor): Batched elelment macrostrain-force matrix, it should be b*24*6 torch.cuda.floatTensor

        Returns:
            [float]: Energy loss
        """
        with torch.no_grad():
            ctx.save_for_backward(input, voxel, Ke_hard, Fe_hard,Ke_soft,Fe_soft)
            ctx.H = H
            ctx.gradient = torch.empty_like(input)
            output = 0

            for i in range(input.shape[0]):
                energy, grad = H.MPE_full(voxel[i], Ke_hard[i], Fe_hard[i], Ke_soft[i], Fe_soft[i], input[i])
                output += energy
                ctx.gradient[i] = grad
    
            output = output/input.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        This is a pattern that is very convenient - at the top of backward
        unpack saved_tensors and initialize all gradients w.r.t. inputs to
        None. Thanks to the fact that additional trailing Nones are
        ignored, the return statement is simple even when the function has
        optional inputs.
        """

        input, voxel, Ke_hard, Fe_hard,Ke_soft,Fe_soft = ctx.saved_variables
        grad = ctx.gradient

        return grad, None, None, None, None,None,None, None, None