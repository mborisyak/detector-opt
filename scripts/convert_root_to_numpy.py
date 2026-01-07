import numpy as np
import uproot
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert ROOT file to NumPy .npz format')
    parser.add_argument('rootfile', type=str, help='Input ROOT file path')
    
    args = parser.parse_args()
    
    rootfile = args.rootfile
    base_name = os.path.splitext(os.path.basename(rootfile))[0]
    out_npz = f"{base_name}.npz"
    
    
    print(f"Reading ROOT file: {rootfile}")
    print(f"Output file: {out_npz}")
    
    with uproot.open(rootfile) as f:
        tree = f["mytree"]
        px  = tree["px"].array(library="np")
        py  = tree["py"].array(library="np")
        pz  = tree["pz"].array(library="np")
        x   = tree["vx"].array(library="np")
        y   = tree["vy"].array(library="np")
        z   = tree["vz"].array(library="np")
        pid = tree["pdgcode"].array(library="np")
        tof = tree["tof"].array(library="np")

    np.savez(out_npz, px=px, py=py, pz=pz, x=x, y=y, z=z, pid=pid, tof=tof)

    print(f"Saved: {out_npz}")
    print("Shapes:", {k: v.shape for k, v in dict(px=px, py=py, pz=pz, x=x, y=y, z=z, pid=pid, tof=tof).items()})

if __name__ == '__main__':
    main()
