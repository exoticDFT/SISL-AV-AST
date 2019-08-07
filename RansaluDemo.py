'''
File: RansaluExample.py
Project: SISL-AV-AST
Created Date: Tuesday August 6th 2019
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
        description='Stanford Adaptive Stress Testing Scenarios for Lincoln Lab'
        ' Demonstration',
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


def initialize_two_cars(world, data, origin, verbose=False):
    # Initialize the actors (two independent cars)
    pos_c = data['car'][0][0:2]
    car1 = ast.initialize_vehicle(world, pos_c, origin, 0.0, 'toyota', verbose)

    pos_p = data['ped'][0][0:2]
    car2 = ast.initialize_vehicle(world, pos_p, origin, 0.0, 'toyota', verbose)

    return (car1, car2)


def visualize_vehicles(
    world,
    data,
    timestep=0.1,
    with_noise=True,
    verbose=False
):
    '''
    Loads in the dataframe containing the first example for AST.
    '''
    car1 = None
    car2 = None

    # Location of origin for this project
    new_origin = np.array([156.0, 110.0])
    origin = carla.Vector3D(156.0, 110.0, 0.0)
    camera_offset = carla.Location(0.0, -20.0, 10.0)

    try:
        util.world.move_spectator(
            world,
            origin + camera_offset,
            carla.Rotation(-25.0, 115.0, 0.0)
        )

        car1, car2 = initialize_two_cars(world, data, new_origin, verbose)

        # Set world to synchronous mode
        ast.set_carla_sync_mode(world, timestep, verbose)
        world.tick()

        # Move the actors
        for i in range(len(data['car'])):
            world.tick()

            # Direct manipulation
            ast.move_actor(
                car1,
                data['car'][i][0:2],
                new_origin,
                verbose
            )
            ast.move_actor(
                car2,
                data['ped'][i][0:2],
                new_origin,
                verbose
            )

            # Visualize the sensor noise
            if with_noise:
                ast.display_sensor_noise(car2, data['ped'][i][4:], timestep)

            time.sleep(timestep)

        world.tick()

        # Set world to non-synchronous mode
        ast.unset_carla_sync_mode(world, verbose)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(5.0)

        if car1:
            car1.destroy()
        if car2:
            car2.destroy()


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

    orig_dt = 0.25
    new_dt = 1.0/60.0

    data_directory = '/home/akoufos/Development/SISL/RansaluExample'

    data = ast.parse_csv(
        os.path.join(data_directory, 'rans_pomdp1-edit.csv'),
        'step',
        args.verbose
    )
    data = ast.interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
    visualize_vehicles(
        carla_world,
        data,
        new_dt,
        False,
        args.verbose
    )


if __name__ == "__main__":
    main()
