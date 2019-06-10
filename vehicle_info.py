'''
File: vehicle_info.py
Project: Carla-Modules
Created Date: Friday June 7th 2019
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

import carla

import math
import random


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(
        d * math.cos(a),
        d * math.sin(a),
        2.0
    ) + vehicle_location
    return carla.Transform(
        location,
        carla.Rotation(yaw=180 + angle, pitch=-15)
    )


def main():
    client = util.client.create(map_name="Town02")
    world = client.get_world()
    spectator = world.get_spectator()
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle')

    location = random.choice(world.get_map().get_spawn_points()).location

    f = open('vehicleInfo.txt', 'w')
    f.write(
        '{:35}{:^13}{:^13}{:^13}\n'.format(
            'Filter String',
            'Length[m]',
            'Width[m]',
            'Height[m]'
        )
    )
    f.write(
        '{:35}{:^13}{:^13}{:^13}\n'.format(
            '----------------',
            '---------',
            '--------',
            '---------'
        )
    )

    for blueprint in vehicle_blueprints:
        transform = carla.Transform(location, carla.Rotation(yaw=-45.0))
        vehicle = world.spawn_actor(blueprint, transform)

        try:

            util.actor.print_info(vehicle)

            f.write(
                '{:35}{:^13.6f}{:^13.6f}{:^13.6f}\n'.format(
                    vehicle.type_id,
                    vehicle.bounding_box.extent.x,
                    vehicle.bounding_box.extent.y,
                    vehicle.bounding_box.extent.z
                )
            )

            # angle = 0
            # while angle < 356:
            #     timestamp = world.wait_for_tick()
            #     angle += timestamp.delta_seconds * 60.0
            #     spectator.set_transform(
            #         get_transform(vehicle.get_location(), angle - 90)
            #     )

        finally:

            vehicle.destroy()

    f.close()


if __name__ == '__main__':

    main()
