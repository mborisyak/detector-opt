from threading import Thread

import multiprocessing as mp
import multiprocessing.connection as mpc
from multiprocessing import Pipe, queues

try:
  from tqdm import tqdm
except ImportError:
  tqdm = None

__all__ = [
  'empty_progress',
  'bar'
]

class empty_progress(object):
  def update(self):
    pass

  def refresh(self, nolock: bool=True):
    pass

  def clear(self):
    pass

  def reset(self, total):
    pass

  def close(self):
    pass

class simple_progress(object):
  def __init__(self, desc: str, total: int, initial: int, _print=print):
    self.desc = desc
    self.total = total
    self.initial = initial

def get_status_bar(starting_epoch: int, epochs: int, train_iterations: int, validation_batches: int, progress):
  if isinstance(progress, queues.Queue):
    return subprocess_bar(starting_epoch, epochs, queue=progress)

  elif isinstance(progress, bool):
    return bar(starting_epoch, epochs, train_iterations, validation_batches, enable=progress)

  else:
    raise ValueError(f'unknown progress = {progress}')

class bar(object):
  def __init__(
    self, starting_epoch: int, epochs: int, train_iterations: int, validation_batches: int, enable: bool | str=True
  ):
    if tqdm is None or enable is False:
      self.main_pb = empty_progress()
      self.train_pb = empty_progress()
      self.validation_pb = empty_progress()
    else:
      self._starting_epoch = starting_epoch
      self._epochs = epochs

      self.main_pb = tqdm(total=epochs, initial=starting_epoch, desc='epochs', disable=not enable, leave=True)
      self.train_pb = tqdm(total=train_iterations, desc='training', disable=not enable, leave=True)
      self.validation_pb = tqdm(total=validation_batches, desc='validation', disable=not enable, leave=True)

  def epoch(self):
    self.main_pb.update()
    # self.main_pb.refresh(nolock=False)

    # self.train_pb.n = 0
    # self.train_pb.refresh(nolock=False)
    # self.validation_pb.n = 0
    # self.validation_pb.refresh(nolock=False)

    self.train_pb.reset()
    self.validation_pb.reset()

  def train_step(self):
    self.train_pb.update()

  def validation_step(self):
    self.validation_pb.update()

  def clear(self):
    self.main_pb.clear()
    self.train_pb.clear()
    self.validation_pb.clear()

  def close(self):
    self.main_pb.close()
    self.train_pb.close()
    self.validation_pb.close()

class subprocess_bar(object):
  def __init__(self, starting_epoch: int, epochs: int, queue):
    self.starting_epoch = starting_epoch
    self.epochs = epochs

    self.queue = queue
    connection = queue.get(block=True)
    self.connection = connection
    self.counter = starting_epoch
    self.connection.send((starting_epoch, epochs))

  def epoch(self):
    self.counter += 1
    self.connection.send(self.counter)

  def train_step(self):
    pass

  def validation_step(self):
    pass

  def clear(self):
    self.counter = self.starting_epoch
    self.connection.send('clear')

  def close(self):
    self.connection.send('close')
    self.connection = None

  def __del__(self):
    if self.connection is not None:
      self.connection.send('close')
      self.connection = None

class main_bar(object):
  def __init__(self, total_tasks: int, workers: int, progress: bool=True, queue=None):
    self.total_tasks = total_tasks
    self.workers = workers

    if progress:
      self.main_pb = tqdm(total=total_tasks, desc='jobs', smoothing=0.05)

      self.queue = queue
      self.connections = {}
      self.bars = []
      for i in range(workers):
        recv, send = Pipe(duplex=False)
        self.queue.put(send)
        self.connections[recv] = i
        self.bars.append(tqdm(total=None))

      self.updating_thread = Thread(target=self.updating, daemon=True)
      self.updating_thread.start()

    else:
      self.main_pb = empty_progress()
      self.queue = None
      self.connections = None
      self.bars = None

  def updating(self):
    while True:
      ready = mpc.wait(self.connections.keys())
      for pipe in ready:
        message = pipe.recv()
        index = self.connections[pipe]
        bar = self.bars[index]

        if isinstance(message, int):
          bar.update(message - bar.n)

        elif isinstance(message, tuple):
          starting_epoch, total_epochs = message
          bar.reset(total=total_epochs)
          bar.n = starting_epoch
          bar.refresh()

        elif message == 'close':
          bar.reset()
          _ = self.connections.pop(pipe)
          recv, send = mp.Pipe(duplex=False)
          self.connections[recv] = index
          self.queue.put(send)
          bar.reset(total=None)

          self.main_pb.update()


  def spawn(self):
    return self.queue

  def update(self):
    self.main_pb.update()

  def close(self):
    self.main_pb.close()
    for bar in self.bars:
      bar.close()