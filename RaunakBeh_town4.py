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
    new_origin = np.array([100.0, 110.0, 0.0])
    origin = carla.Vector3D(100.0, 110.0, 0.0)
    camera_offset = carla.Location(0.0, -20.0, 120.0)
    myblueprint = world.get_blueprint_library().filter("vehicle.toyota.*")[0]
    car = None
    car2 = None
    car3 = None

    try:
        #util.world.move_spectator(world,origin + camera_offset,carla.Rotation(-25.0, 115.0, 0.0))

        spawnpoints = world.get_map().get_spawn_points()

        util.world.move_spectator(world,spawnpoints[299].location+carla.Location(0.0,0.0,50.0),carla.Rotation(0.0,0.0,0.0))
        # TODO: randomize spawnpoints

#        for waypoint in spawnpoints:
#            print(waypoint.location)

        # To visualize spawnpoints
        util.world.draw_spawn_points(world,timeout=20.0)

        #pos = [-20.,0.]
        #car = ast.initialize_vehicle(world,pos,new_origin,0.0,'toyota',color,verbose)
        car = util.actor.initialize(world,myblueprint,transform=spawnpoints[299],verbose=False)
        agent = BehaviorAgent(car,behavior='aggressive')
        dest_wp_1 = spawnpoints[2].location
        print("dest for car 1 and car2 is",dest_wp_1)
        agent.set_destination(car.get_location(),dest_wp_1,clean=True)
        #waypoints = [i[0] for i in agent.get_local_planner().waypoints_queue]
        #misc.draw_waypoints(car.get_world(),waypoints,timeout=60.0) #disp waypoints for 60 sec
        time.sleep(1.0)

#        pos2 = [-30.,0.]
#        car2 = ast.initialize_vehicle(world,pos2,new_origin,0.0,'toyota',color,verbose)  
        car2 = util.actor.initialize(world,myblueprint,transform=spawnpoints[303],verbose=False)
        agent2 = BehaviorAgent(car2,behavior='aggressive')
        agent2.set_destination(car2.get_location(),dest_wp_1,clean=True)
        time.sleep(1.0)

#        pos3 = [-20.,5.]
#        car3 = ast.initialize_vehicle(world,pos3,new_origin,0.0,'toyota',color,verbose)
#        agent3 = BehaviorAgent(car3,behavior='aggressive')
        #dest_wp_3 = spawnpoints[2].location
#        agent3.set_destination(car3.get_location(),dest_wp_1,clean=True)
#        print("dest for car 3 is",dest_wp_3)
        #print(car3.get_location())
#        time.sleep(1.0)


        for waypoint in agent._local_planner.waypoints_queue:
            world.debug.draw_string(waypoint[0].transform.location,'o',draw_shadow=False,color=carla.Color(r=255, g=255, b=255), life_time=10.0,persistent_lines=True)

        while True:
            world.wait_for_tick()
            agent.update_information()
            control = agent.run_step()
            #print("agent control is",control)
            agent.vehicle.apply_control(control)

            agent2.update_information()
            control2 = agent2.run_step()
            agent2.vehicle.apply_control(control2)

#            agent3.update_information()
#            control3 = agent3.run_step()
#            agent3.vehicle.apply_control(control3)
        
            if len(agent.get_local_planner().waypoints_queue) < 21:
                agent.reroute(spawnpoints)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(2.0)
        car.destroy()
        car2.destroy()
#        car3.destroy()

def main():
    args = parse_arguments()

    carla_client = util.client.create(
        args.host,
        args.port,
        args.timeout,
        'Town04'
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
