import os
import numpy as np

def test_event_files():
  import uproot

  rootfile = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data', 'clean.root'
  )

  tree_name = "mytree"
  px_name = "px"
  py_name = "py"
  pz_name = "pz"
  x_name = "vx"
  y_name = "vy"
  z_name = "vz"
  pid_name = "pdgcode"

  n = 10

  with uproot.open(rootfile) as file:
    tree = file[tree_name]
    event_size = tree[event_size_name].array(library="np")
    HNL_vx = tree[HNL_vx_name].array(library="np")
    px = tree[px_name].array(library="np")
    py = tree[py_name].array(library="np")
    pz = tree[pz_name].array(library="np")
    x = tree[x_name].array(library="np")
    y = tree[y_name].array(library="np")
    z = tree[z_name].array(library="np")
    pid = tree[pid_name].array(library="np")

  import matplotlib.pyplot as plt

  fig = plt.figure()
  axes = fig.subplots()
  axes.scatter(px, py)

  plt.show()