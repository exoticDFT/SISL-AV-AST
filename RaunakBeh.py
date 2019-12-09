import util.actor
import util.client
import util.world
import ast_test as ast

import carla

import numpy as np

import argparse
import os
import time

from agents.navigation.behavior.behavior_agent import BehaviorAgent
from agents.tools import misc

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

def move_vehicles(world,color,verbose=False):
    '''
    Behavior agent move
    '''
    print("beh agent move called")
    # Location of origin for this project
    new_origin = np.array([156.0, 110.0, 0.0])
    origin = carla.Vector3D(156.0, 110.0, 0.0)
    camera_offset = carla.Location(0.0, -20.0, 20.0)

    car = None
    try:
        util.world.move_spectator(
            world,
            origin + camera_offset,
            carla.Rotation(-25.0, 115.0, 0.0)
        )
        spawnpoints = world.get_map().get_spawn_points()
        pos = [-20.,0.]
        car = ast.initialize_vehicle(world,pos,new_origin,0.0,'toyota',color,verbose)
        agent = BehaviorAgent(car,behavior='aggressive')
        agent.set_destination(car.get_location(),spawnpoints[0].location,clean=True)
        #waypoints = [i[0] for i in agent.get_local_planner().waypoints_queue]
        #misc.draw_waypoints(car.get_world(),waypoints,timeout=60.0) #disp waypoints for 60 sec

        while True:
            world.wait_for_tick()
            agent.update_information()
            control = agent.run_step()
            agent.vehicle.apply_control(control)
        
            if len(agent.get_local_planner().waypoints_queue) < 21:
                agent.reroute(spawnpoints)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(2.0)
        car.destroy()

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

    move_vehicles(carla_world,'23,51,243',args.verbose)

if __name__ == "__main__":
    main()
