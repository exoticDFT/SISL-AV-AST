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

def place_vehicles(world,data,origin,color='',verbose=False):
    print("place_vehicles has been called")
    num_veh = int(data.shape[1]/2)
    car_list = []
    for i in range(num_veh):
        pos = data[0][i*2:(i+1)*2]
        car_list.append(ast.initialize_vehicle(world,pos,origin,0.0,'toyota',color,verbose))

    return car_list

def place_vehicles_higher(world,data,origin,color='',verbose=False):
    print("place_vehicles_higher has been called")
    num_veh = int(data.shape[1]/2)
    car_list = []
    for i in range(num_veh):
        pos = data[0][i*2:(i+1)*2]
        car_list.append(ast.initialize_vehicle_higher(world,pos,origin,0.0,'toyota',color,verbose))

    return car_list

def move_vehicles(world,data,timestep=0.1,color='',verbose=False):
    '''
    Move vehicels according to provided trajectory in data.
    '''
    print("move_vehicles has been called")
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

        car_list = place_vehicles(world, data, new_origin, color, verbose)

        # Set world to synchronous mode
        ast.set_carla_sync_mode(world, timestep, verbose)
        world.tick()

        # Move the actors
        for i in range(len(data)):
            print("iteration = ",i)
            world.tick()

            for j in range(len(car_list)):
                car = car_list[j]
                move_actor(car,data[i][(j)*2:(j+1)*2],new_origin+[0.0,0.0,0.5],verbose)

            time.sleep(timestep)

        world.tick()

        # Set world to non-synchronous mode
        ast.unset_carla_sync_mode(world, verbose)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(2.0)

        for k in range(len(car_list)):
            car2destroy = car_list[k]
            car2destroy.destroy()

def move_vehicles_true_imit(world,data_true,data_imit,timestep=0.1,color_true='',color_imit='',verbose=False):
    '''
    Move ground truth and imitated vehicles according to data_true and data_imit.
    '''
    print("move_vehicles_true_imit has been called")
    # Location of origin for this project
    new_origin = np.array([-180.0, 108.0, 0.0])
    origin = carla.Vector3D(80.0, 110.0, 0.0)
    camera_offset = carla.Location(0.0, -20.0, 20.0)

    try:
        util.world.move_spectator(
            world,
            origin + camera_offset,
            carla.Rotation(-25.0, 115.0, 0.0)
        )

        car_list_true = place_vehicles(world, data_true, new_origin, color_true, verbose)
        car_list_imit = place_vehicles_higher(world, data_imit, new_origin, color_imit, verbose)

        # Set world to synchronous mode
        ast.set_carla_sync_mode(world, timestep, verbose)
        world.tick()

        # Move the actors
        for i in range(len(data_true)):
            print("iteration = ",i)
            world.tick()

            for j in range(len(car_list_true)):
                car_true = car_list_true[j]
                move_actor(car_true,data_true[i][(j)*2:(j+1)*2],new_origin+[0.0,0.0,0.5],verbose)

                car_imit = car_list_imit[j]
                move_actor(car_imit,data_imit[i][(j)*2:(j+1)*2],new_origin+[0.0,0.0,0.5],verbose)

            time.sleep(timestep)

        world.tick()

        # Set world to non-synchronous mode
        ast.unset_carla_sync_mode(world, verbose)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(2.0)

        for k in range(len(car_list_true)):
            car2destroy = car_list_true[k]
            car2destroy_imit = car_list_imit[k]
            car2destroy.destroy()
            car2destroy_imit.destroy()

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

    data_true = np.loadtxt(os.path.join(data_directory, 'ground_truth.csv'),delimiter=',')
    data_imit = np.loadtxt(os.path.join(data_directory, 'imitation.csv'),delimiter=',')
    #move_vehicles(carla_world,data_imit,0.02,'255,255,255',args.verbose)
    move_vehicles_true_imit(carla_world,data_true,data_imit,0.02,'255,255,255','23,51,243',args.verbose)


if __name__ == "__main__":
    main()