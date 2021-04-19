import time
import argparse
import pprint as pp
import os

import pandas as pd
import numpy as np
from concorde.tsp import TSPSolver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--node_dim", type=int, default=2)
    parser.add_argument("--filename", type=str, default=None)
    opts = parser.parse_args()
    
    if opts.filename is None:
        opts.filename = f"tsp{opts.num_nodes}_concorde_new.txt"
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    set_nodes_coord = np.random.random([opts.num_samples, opts.num_nodes, opts.node_dim])
    scaled_set_nodes_coord = 1000*set_nodes_coord
    with open(opts.filename, "w") as f:
        start_time = time.time()
        for i, nodes_coord in enumerate(scaled_set_nodes_coord):
            solver = TSPSolver.from_data(nodes_coord[:,0], nodes_coord[:,1], norm="EUC_2D")
            solution = solver.solve()
            f.write( " ".join( str(x)+str(" ")+str(y) for x,y in set_nodes_coord[i]) )
            f.write( str(" ") + str('output') + str(" ") )
            f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour) )
            f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
            f.write( "\n" )
        end_time = time.time() - start_time
    
    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.num_nodes}.")
    print(f"Total time: {end_time/3600:.1f}h")
    print(f"Average time: {(end_time/3600)/opts.num_samples:.1f}h")
