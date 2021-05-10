import numpy as np
from functools import partial
from scipy.misc import derivative
from scipy.optimize import fsolve
import argparse
import sys

from functions import analytical_jacobian, f1, f2, starting_point

def create_f_system(x, f1, f2):
    return [f1(*x), f2(*x)]

def create_num_jacobian(f1, f2):
    def num_jacobian(x, f1, f2):
        j11 = derivative(partial(f1, x2=x[1]), x[0], dx=1e-1)
        j12 = derivative(partial(f1, x[0]), x[1], dx=1e-1)
        j21 = derivative(partial(f2, x2=x[1]), x[0], dx=1e-1)
        j22 = derivative(partial(f2, x[0]), x[1], dx=1e-1)
        return [[j11, j12], [j21, j22]]
    return partial(num_jacobian, f1=f1, f2=f2)

def run(f1, f2, x0, eps=0.00001, is_num=True, jac_fn=None, verbose=False):

    f_system = partial(create_f_system, f1=f1, f2=f2)

    if is_num:
        jac_fn = create_num_jacobian(f1, f2)
    if jac_fn is None:
        raise ValueError('Provide jacobian function to solve system by analytical method')
    def approx_func(x):
        return x - np.dot(np.linalg.inv(jac_fn(x)), f_system(x))

    if verbose:
        print(f'Function at starting point: f({x0}) = {f_system(x0)}')
        print(f'Jacobian at starting point: J({x0}) = {jac_fn(x0)}')
        print(f'Invariated Jacobian at starting point: J^-1({x0}) = {np.linalg.inv(jac_fn(x0))}')

    delta = 1
    iter = 0
    x = x0
    while delta > eps:
        x_last = x
        x = approx_func(x)
        delta = np.linalg.norm(x - x_last)
        print(f'Iter {iter} x={x} delta={delta}')
        iter += 1

    print(f'Result: x={x} f(x)={f_system(x)}')


def built_in_solve(f1, f2, x0):
    f_system = partial(create_f_system, f1=f1, f2=f2)
    result = fsolve(f_system, starting_point)
    print(f'Result: x={result} f(x)={f_system(result)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='num', help='path to config', choices=['fsolve', 'analytical', 'num'])
    args = parser.parse_args(args=sys.argv[1:])
    if args.method == 'fsolve':
        print('Solving problem using built-in function')
        f_system = partial(create_f_system, f1=f1, f2=f2)
        built_in_solve(f1, f2, starting_point, )
    elif args.method == 'num':
        print('Solving problem using numerical calculations')
        run(f1, f2, starting_point)
    else:
        print('Solving problem using human brain power (analytical derivation)')
        run(f1, f2, starting_point, is_num=False, jac_fn=analytical_jacobian, verbose=False)
