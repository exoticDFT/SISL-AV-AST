'''
File: ast_test.py
Project: Carla-Modules
Created Date: Thursday June 6th 2019
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

import carla
import numpy as np
import pandas
from scipy import interpolate

import argparse
import time

# Init global simulation related variables
carla_world = None


def parse_arguments():
    '''
    The argument parser used for the ast script.
    '''
    argparser = argparse.ArgumentParser(
        description='AST Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='The ip address of the host server'
    )
    argparser.add_argument(
        '--filename',
        '-f',
        metavar='F',
        help='The filename containing the movement for the actors'
    )
    argparser.add_argument(
        '--map',
        '-m',
        metavar='M',
        default='Town03',
        help='The map name the Carla server should load'
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


def parse_csv(filename, index_column, verbose=False):
    df = pandas.read_csv(filename, index_col=index_column)

    df.columns = df.columns.str.strip().str.lower().str.replace(
        ' ',
        '_'
    ).str.replace(
        '(',
        ''
    ).str.replace(
        ')',
        ''
    )

    if verbose:
        print('Parsed dataframe information:')
        print('-----------------------------')
        print('First 5 rows (head)\n', df.head())
        print('Dataframe keys:\n', df.keys())

    car = df[[
        'x_car',
        'y_car',
        'v_x_car',
        'v_y_car'
    ]].to_numpy()
    ped = df[[
        'x_ped_0',
        'y_ped_0',
        'v_x_ped_0',
        'v_y_ped_0',
        'noise_x_0',
        'noise_y_0'
    ]].to_numpy()

    parsed = {'car': car, 'ped': ped}

    return parsed


def interpolate_data(data, orig_step=0.1, new_step=1.0/60.0, verbose=False):
    stop = len(data)*orig_step

    if verbose:
        print('Stop:', stop)

    t = np.arange(0.0, stop, orig_step)
    pos = data[:, 0:2].transpose()
    vel = data[:, 2:4].transpose()

    has_error = data.shape[1] == 6

    if has_error:
        err = data[:, 4:6].transpose()
    else:
        err = np.full(pos.shape, 0.0)

    if verbose:
        print('Input to interpolate')
        print('--------------------')
        print('Time:\n', len(t), t)
        print('Position:\n', len(pos[0]), pos)
        print('Velocity:\n', len(vel[0]), vel)
        print('Sensor Noise:\n', len(err[0]), err)

    f_pos = interpolate.interp1d(t, pos)
    f_vel = interpolate.interp1d(t, vel)
    f_err = interpolate.interp1d(t, err)

    new_t = np.arange(0.0, stop - orig_step, new_step)

    new_pos = f_pos(new_t)
    new_vel = f_vel(new_t)
    new_err = f_err(new_t)
    new_data = np.concatenate((new_pos, new_vel, new_err))

    if verbose:
        print('Interpolated data')
        print('-----------------')
        print('New Time:\n', len(new_t), new_t)
        print('New Data:\n', new_data)

    return new_data.transpose()


def interpolate_car_and_ped(
    data,
    orig_step=0.1,
    new_step=1.0/60.0,
    verbose=False
):
    car = interpolate_data(data['car'], orig_step, new_step, verbose)
    ped = interpolate_data(data['ped'], orig_step, new_step, verbose)

    output = {'car': car, 'ped': ped}

    if verbose:
        print('Final data after interpolation')
        print('------------------------------')
        print('Car:\n', data['car'])
        print('Pedestrian:\n', data['ped'])

    return output


def initialize_vehicle_and_walker(world, data, origin, verbose=False):
    # Initialize the actors (car and pedestrian)
    pos_c = data['car'][0][0:2]
    car = init_vehicle(pos_c, origin, 0.0, verbose=verbose)

    pos_p = data['ped'][0][0:2]
    ped = init_walker(pos_p, origin, 270.0, verbose=verbose)

    # Create pedestrian controller
    ped_control = create_ped_control(data['ped'][0][2:4])
    ped.apply_control(ped_control)

    return (car, ped, ped_control)


def set_carla_sync_mode(world, timestep=0.1, verbose=False):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = timestep

    if verbose:
        print('Carla synchronous mode: ON')

    world.apply_settings(settings)


def unset_carla_sync_mode(world, verbose=False):
    settings = world.get_settings()
    settings.synchronous_mode = False

    if verbose:
        print('Carla synchronous mode: ON')

    world.apply_settings(settings)


def visualize_vehicle_and_walker(
    world,
    data,
    timestep=0.1,
    with_noise=True,
    verbose=False
):
    '''
    Loads in the dataframe containing the first example for AST.
    '''
    car = None
    ped = None

    # Location of origin for this project
    new_origin = np.array([156.0, 110.0])
    origin = carla.Vector3D(156.0, 110.0, 0.0)
    camera_offset = carla.Location(0.0, -20.0, 10.0)

    try:
        # Set world to synchronous mode
        set_carla_sync_mode(world, timestep, verbose)

        util.world.move_spectator(
            world,
            origin + camera_offset,
            carla.Rotation(-25.0, 115.0, 0.0)
        )

        car, ped, ped_control = initialize_vehicle_and_walker(
            world,
            data,
            new_origin,
            verbose
        )

        world.tick()

        # Move the actors
        for i in range(len(data['car'])):
            world.tick()

            # Direct manipulation
            move_actor(
                car,
                data['car'][i][0:2],
                new_origin,
                verbose
            )

            # Control-based
            apply_ped_control(ped, data['ped'][i][2:4], verbose)

            # Visualize the sensor noise
            if with_noise:
                display_sensor_noise(ped, data['ped'][i][4:], timestep)

            time.sleep(timestep)

        apply_ped_control(ped, [0.0, 0.0], verbose)

        world.tick()

        # Set world to non-synchronous mode
        unset_carla_sync_mode(world, verbose)

    finally:
        # Wait for a bit before destroying the actors
        time.sleep(5.0)

        if car:
            car.destroy()
        if ped:
            ped.destroy()


def display_sensor_noise(ped, pos, timestep=0.1):
    util.actor.draw_boundingbox(
        ped,
        life_time=timestep,
        color=carla.Color(255, 255, 255),
        thickness=0.05,
        offset=carla.Location(pos[0], pos[1], 0.0)
    )


def apply_ped_control(actor, velocity, verbose=False):
    control = actor.get_control()
    speed = np.linalg.norm(velocity)
    direction = carla.Vector3D(velocity[0], -velocity[1], 0.0)
    control.direction = direction
    control.speed = speed

    if verbose:
        print('Pedestrain velocity:', velocity)
        print('Pedestrain speed:', control.speed)
        print('Pedestrain direction:', control.direction)
        util.actor.print_info(actor)
        util.actor.draw_boundingbox(
            actor,
            life_time=0.05,
            thickness=0.02
        )

    actor.apply_control(control)


def create_ped_control(velocity):
    speed = np.linalg.norm(velocity)
    direction = carla.Vector3D(velocity[0], -velocity[1], 0.0)
    control = carla.WalkerControl(direction, speed)
    return control


def init_vehicle(pos, offset, heading, model='lincoln', verbose=False):
    '''
    Initializes a Carla actor with the provided data and returns the created
    actor.
    '''
    position = carla.Vector3D(
        pos[0] + offset[0],
        -pos[1] + offset[1],
        0.5
    )
    rotation = carla.Rotation(0.0, heading, 0.0)

    blueprints = carla_world.get_blueprint_library().filter(
        'vehicle.' + model + '.*'
    )

    bp = util.actor.create_random_blueprint(blueprints)

    actor = util.actor.initialize(
        carla_world,
        bp,
        position,
        rotation,
        verbose=verbose
    )

    if actor:
        actor.set_simulate_physics(False)

    return actor


def init_walker(pos, offset, heading, verbose=False):
    '''
    Initializes a Carla actor with the provided data and returns the created
    actor.
    '''
    position = carla.Vector3D(
        pos[0] + offset[0],
        -pos[1] + offset[1],
        1.3
    )
    rotation = carla.Rotation(0.0, heading, 0.0)

    blueprints = carla_world.get_blueprint_library().filter('walker')
    ped_bp = util.actor.create_random_blueprint(blueprints)

    actor = util.actor.initialize(
        carla_world,
        ped_bp,
        position,
        rotation,
        verbose=verbose
    )

    if actor:
        actor.set_simulate_physics(False)

    return actor


def move_actor(actor, pos, offest, verbose):
    '''
    Moves a given Carla actor according to the provided data and timestep.
    '''
    if actor:
        position = carla.Vector3D(
            pos[0] + offest[0],
            -pos[1] + offest[1],
            0.25
        )
        actor.set_location(position)

        if verbose:
            util.actor.print_info(actor)
            util.actor.draw_boundingbox(
                actor,
                life_time=0.05,
                thickness=0.02,
                offset=carla.Location(0.0, 0.0, -0.2)
            )


def main():
    args = parse_arguments()

    global carla_world

    carla_client = util.client.create(
        args.host,
        args.port,
        args.timeout,
        args.map
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

    # First lets read in the csv file
    if args.filename:
        orig_dt = 0.1
        new_dt = 1.0/60.0
        data = parse_csv(args.filename, 'step', args.verbose)
        data = interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
        visualize_vehicle_and_walker(
            carla_world,
            data,
            new_dt,
            False,
            args.verbose
        )
    else:
        orig_dt = 0.1
        new_dt = 1.0/20.0

        # AST 1
        data = parse_csv(
            "/home/akoufos/Development/SISL/LincolnLabExample1/sample_trajectory.csv",
            'step',
            args.verbose
        )
        data = interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
        visualize_vehicle_and_walker(
            carla_world,
            data,
            new_dt,
            False,
            args.verbose
        )

        # AST 2
        data = parse_csv(
            "/home/akoufos/Development/SISL/LincolnLabExample1/new_Traj.csv",
            'step',
            args.verbose
        )
        data = interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
        visualize_vehicle_and_walker(
            carla_world,
            data,
            new_dt,
            False,
            args.verbose
        )

        # AST 3
        data = parse_csv(
            "/home/akoufos/Development/SISL/LincolnLabExample1/no_crash_traj.csv",
            'step',
            args.verbose
        )
        data = interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
        visualize_vehicle_and_walker(
            carla_world,
            data,
            new_dt,
            False,
            args.verbose
        )

        # AST 4
        data = parse_csv(
            "/home/akoufos/Development/SISL/LincolnLabExample1/ped_fault_traj.csv",
            'step',
            args.verbose
        )
        data = interpolate_car_and_ped(data, orig_dt, new_dt, args.verbose)
        visualize_vehicle_and_walker(
            carla_world,
            data,
            new_dt,
            False,
            args.verbose
        )


if __name__ == "__main__":
    main()
