'''
File: PeterDemo.py
Project: SISL-AV-AST
Created Date: Tuesday October 14th 2019
Author: Alexander Koufos, PhD <akoufos@stanford.edu>
-----
Last Modified:
Modified By:
-----
Copyright (c) 2019 Stanford Intelligent Systems Lab - Stanford University
License: MIT License
'''
import util.actor
import util.client
import util.world
import ast_test as ast

import carla
import numpy as np
import pandas
from scipy import interpolate

import argparse
import os
import time


def parse_arguments():
    '''
    The argument parser used for the ast script.
    '''
    argparser = argparse.ArgumentParser(
        description='Peter Du - Adaptive Stress Testing Scenarios.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='The ip address of the host server'
    )
    argparser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port used for listening'
    )
    argparser.add_argument(
        '-t',
        '--timeout',
        metavar='T',
        default=3.0,
        type=float,
        help='Timeout, in seconds, of the Carla client when contacting server'
    )
    argparser.add_argument(
        '-v',
        '--verbose',
        dest='verbose',
        default=False,
        action='store_true',
        help='Boolean to toggle the output of verbose information'
    )

    args = argparser.parse_args()
    args.description = argparser.description

    return args


def main():
    args = parse_arguments()

    carla_client = util.client.create(
        args.host,
        args.port,
        args.timeout,
        'Town02'
    )
    carla_world = carla_client.get_world()

    weather = carla.WeatherParameters(
            cloudyness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=130.0,
            sun_altitude_angle=68.0)
    carla_world.set_weather(weather)

    orig_dt = 0.1
    new_dt = 1.0/25.0

    data_directory = '/home/akoufos/Development/SISL/PeterExample'

    # Peter 1
    data = ast.parse_csv(
        os.path.join(data_directory, 'trajectory_1.csv'),
        'step',
        args.verbose
    )
    data = ast.interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
    ast.visualize_vehicle_and_walker(
        carla_world,
        data,
        new_dt,
        True,
        args.verbose
    )

    # Peter 2
    data = ast.parse_csv(
        os.path.join(data_directory, 'trajectory_2.csv'),
        'step',
        args.verbose
    )
    data = ast.interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
    ast.visualize_vehicle_and_walker(
        carla_world,
        data,
        new_dt,
        True,
        args.verbose
    )

    # Peter 3
    data = ast.parse_csv(
        os.path.join(data_directory, 'trajectory_3.csv'),
        'step',
        args.verbose
    )
    data = ast.interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
    ast.visualize_vehicle_and_walker(
        carla_world,
        data,
        new_dt,
        True,
        args.verbose
    )


if __name__ == "__main__":
    main()
