"""Single-device mean-loss backward with exactly one accumulation normalization."""
def mean_training_step(self, model, inputs, num_items_in_batch=None):
    model.train()
    if hasattr(self.optimizer, 'train'):
        self.optimizer.train()
    inputs = self._prepare_inputs(inputs)
    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
    # Accelerator.backward divides by configured GAS. Compensate only to use
    # the ACTUAL window size (important for the last partial window).
    window = self.current_gradient_accumulation_steps
    self.accelerator.backward(loss * (self.accelerator.gradient_accumulation_steps / window))
    return loss.detach() / window
