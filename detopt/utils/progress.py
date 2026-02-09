try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

__all__ = [
  'status_bar'
]

class empty_progress(object):
  def update(self):
    pass

  def refresh(self):
    pass

  def clear(self):
    pass

class status_bar(object):
  def __init__(self, disable=False):
    if tqdm is None:
      self.main_pb = empty_progress()
      self.train_pb = empty_progress()
      self.validation_pb = empty_progress()
    else:
      self.main_pb = tqdm(total=None, desc='epochs', disable=disable, leave=True)
      self.train_pb = tqdm(total=None, desc='training', disable=disable, leave=True)
      self.validation_pb = tqdm(total=None, desc='validation', disable=disable, leave=True)

  def epochs(self, start, finish=None):
    if finish is None:
      finish = start
      start = 0

    self.main_pb.reset(total=finish - start)
    for i in range(start, finish):
      yield i
      self.main_pb.update()
    self.main_pb.refresh(nolock=True)

  def training(self, steps):
    self.train_pb.reset(total=steps)

    for j in range(steps):
      yield j
      self.train_pb.update()
    self.train_pb.refresh(nolock=True)

  def validation(self, steps):
    self.validation_pb.reset(total=steps)

    for k in range(steps):
      yield k
      self.validation_pb.update()
    self.validation_pb.refresh(nolock=True)

  def clear(self):
    self.main_pb.clear()
    self.train_pb.clear()
    self.validation_pb.clear()