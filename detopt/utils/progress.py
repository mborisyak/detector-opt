from tqdm import tqdm

__all__ = [
  'status_bar'
]

class status_bar(object):
  def __init__(self, epochs, training_steps, validation_steps, disable=False):
    self.epochs_pb = tqdm(total=epochs, desc='epochs', disable=disable)
    self.iter_pb = tqdm(total=training_steps, desc='training', disable=disable)
    self.val_pb = tqdm(total=validation_steps, desc='validation', disable=disable)

    self._epochs = epochs
    self._steps = training_steps
    self._validation_steps = validation_steps

  def epochs(self):
    self.epochs_pb.reset(self._epochs)
    for i in range(self._epochs):
      yield i
      self.epochs_pb.update()

  def training(self):
    self.iter_pb.reset(self._steps)
    for j in range(self._steps):
      yield j
      self.iter_pb.update()

    self.iter_pb.refresh(nolock=True)

  def validation(self):
    self.val_pb.reset(self._validation_steps)
    for j in range(self._validation_steps):
      yield j
      self.val_pb.update()

    self.val_pb.refresh(nolock=True)