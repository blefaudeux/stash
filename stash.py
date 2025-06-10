# We'll write a module wrapper which moves the activations to the CPU
# while a delayed reward is being computed, then move it back to GPU when timing is appropriate.
import torch


# Using naively the save_on_cpu context manager to wrap the module for now
# Ideally we should have a way to stash until a given reward is received / verifiers did their job
# So we would have a mechanism for a multiple stash, and a way to retrieve the stashed activations
class StashWrap(torch.nn.Module):
  def __init__(self, module):
    super(StashWrap, self).__init__()
    self.module = module

  def forward(self, x):
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
      y = self.module(x)

    return y
