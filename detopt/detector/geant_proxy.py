import numpy as np

from .common import Detector

# np.savez(
#     'randomized.npz',
#     positions=positions[:event_i],
#     mask=mask[:event_i],
#     tdc=tdc[:event_i],
#     hnl_momenta=hnl_momenta[:event_i],
#     hnl_decay=hnl_decay[:event_i],
#     hnl_mass=hnl_mass[:event_i],
#     reconstructed_momenta=reconstructed_momenta[:event_i],
#     reconstructed=reconstructed[:event_i],
#     design=design[:event_i]
# )

class GeantProxy(Detector):
  def __init__(self, path, stations: int=4, views: int=4, layers: int=2, tubes: int=317):
    data = np.load(path)
    self.positions = data['positions']
    self.mask = data['mask']
    self.tdc = data['tdc']
    self.hnl_momenta = data['hnl_momenta']
    self.hnl_decay = data['hnl_decay']
    self.hnl_mass = data['hnl_mass']
    self.reconstructed_momenta = data['reconstructed_momenta']
    self.reconstructed = data['reconstructed']
    self.design = data['design']

    self.stations = stations
    self.views = views
    self.layers = layers
    self.tubes = tubes

  def design_shape(self):
    return self.design.shape[1:]

  def output_shape(self):
    return self.stations, self.views * self.layers, self.tubes

  def target_shape(self):
    return (3, )

  def ground_truth_shape(self):
    return (6, )

  def __call__(self, seed: int, configuration: np.ndarray):