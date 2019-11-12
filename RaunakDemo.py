'''
File: RaunakDemo.py
Project: SISL-AV-AST
Created Date: Thursday November 6th 2019
Author: Raunak Bhattacharyya <raunakbh@stanford.edu>
-----
Last Modified: Tuesday Nov 12th 2019, added option to color vehicles
Modified By: Raunak Bhattacharyya <raunakbh@stanford.edu>
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


def initialize_three_cars(world, data, origin, color='',verbose=False):
    # Initialize the actors (two independent cars)
    pos_1 = data[0][0:2]
    print("Initialize 3 cars says: pos1 = ",pos_1)
    car1 = ast.initialize_vehicle(world, pos_1, origin, 0.0, 'toyota', color, verbose)

    pos_2 = data[0][2:4]
    car2 = ast.initialize_vehicle(world, pos_2, origin, 0.0, 'toyota', color, verbose)
    print("Initialize 3 cars says: pos2 = ",pos_2)

    pos_3 = data[0][4:6]
    car3 = ast.initialize_vehicle(world, pos_3, origin, 0.0, 'toyota', color, verbose)
    print("Initialize 3 cars says: pos3 = ",pos_3)

    return (car1, car2, car3)


def move_actor(actor, pos, offset, verbose):
    '''
    Moves a given Carla actor according to the provided data and timestep.
    '''
    if actor:
        position = carla.Vector3D(
            pos[0] + offset[0],
            -pos[1] + offset[1],
            offset[2]
        )
        rotation = carla.Rotation(yaw=0.0)
        transform = carla.Transform(position, rotation)
        actor.set_transform(transform)

        if verbose:
            util.actor.print_info(actor)
            util.actor.draw_boundingbox(
                actor,
                life_time=0.05,
                thickness=0.02,
                offset=carla.Location(0.0, 0.0, -0.2)
            )


def visualize_vehicles(
    world,
    data,
    timestep=0.1,
    color='',
    verbose=False
):
    '''
    Loads in the dataframe containing the first example for AST.
    '''
    car1 = None
    car2 = None

    # Location of origin for this project
    new_origin = np.array([-180.0, 110.0, 0.0])
    origin = carla.Vector3D(80.0, 110.0, 0.0)
    camera_offset = carla.Location(0.0, -20.0, 20.0)

    try:
        util.world.move_spectator(
            world,
            origin + camera_offset,
            carla.Rotation(-25.0, 115.0, 0.0)
        )

        car1, car2, car3 = initialize_three_cars(world, data, new_origin, color, verbose)

        # Set world to synchronous mode
        ast.set_carla_sync_mode(world, timestep, verbose)
        world.tick()

        # Move the actors
        for i in range(len(data)):
            print("iteration = ",i)
            world.tick()

            # Direct manipulation
            move_actor(car1,data[i][0:2],new_origin + [0.0, 0.0, 0.5],verbose)
            move_actor(car2,data[i][2:4],new_origin + [0.0, 0.0, 0.5],verbose)
            move_actor(car3,data[i][4:6],new_origin + [0.0, 0.0, 0.5],verbose)

            time.sleep(timestep)

        world.tick()

        # Set world to non-synchronous mode
        ast.unset_carla_sync_mode(world, verbose)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(2.0)

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

    data_directory = '/scratch/SISL-AV-AST'

    data = np.loadtxt(os.path.join(data_directory, 'ground_truth.csv'),delimiter=',')
    data_imitation = np.loadtxt(os.path.join(data_directory, 'imitation.csv'),delimiter=',')
    visualize_vehicles(carla_world,data,0.02,'23,51,243',args.verbose)


if __name__ == "__main__":
    main()
