#region modules
from argparse import ArgumentParser
import argparse 
from paraprof.scaling.strong_scaling import StrongScaling
from paraprof.scaling.weak_scaling import WeakScaling
import os 
#endregion

#region variables
#endregion

#region functions
def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        help='Can be strong-scaling or weak-scaling.',
    )

    parser.add_argument(
        '--openmp',
        action='store_true',
        default=False,
        help='Choose openmp scaling mode.'
    )

    parser.add_argument(
        '--mpi',
        action='store_true',
        default=False,
        help='Choose mpi scaling mode.'
    )

    parser.add_argument(
        '--ntasks',
        nargs='+',
        type=int,
        help='List of total MPI tasks',
        default=[1],
    )

    parser.add_argument(
        '--nthreads',
        nargs='+',
        type=int,
        default=[2],
        help='List of openmp threads per task',
    )

    parser.add_argument(
        '--problem_sizes',
        nargs='+',
        type=int,
        help='List of problem sizes',
    )

    parser.add_argument(
        '--program',
        type=str,
        help='Can be strong-scaling, weak-scaling, or speedup',
    )

    parser.add_argument(
        '--mpiexec',
        type=str,
        help='mpi scheduler. Default is srun.',
        default='srun',
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot the collected data',
        default=False,
    )


    args = parser.parse_args()

    if args.type is not None:
        match args.type:
            case 'strong-scaling':
                if args.plot:
                    StrongScaling(args).plot()
                    return 
                StrongScaling(args).write()
            case 'weak-scaling':
                if args.plot:
                    WeakScaling(args).plot()
                    return 
                WeakScaling(args).write()
            case _:
                NotImplementedError('--type has to be strong-scaling or weak-scaling.')
        os.system('chmod u+x ./*.sh')

#endregion

#region classes
#endregion