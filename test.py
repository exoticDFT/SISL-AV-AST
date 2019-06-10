import sensors.cameras

import carla

import random
import time


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)

    world = client.get_world()

    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = random.choice(
        world.get_blueprint_library().filter('vehicle.bmw.*')
    )

    vehicle = world.spawn_actor(
        vehicle_bp,
        spawn_points[1]
    )

    ped_bp = random.choice(
        world.get_blueprint_library().filter('walker')
    )

    ped = world.spawn_actor(
        ped_bp,
        carla.Transform(
            carla.Location(75.5166, 12.80842, 0.8431),
            carla.Rotation(0.0, 270.0, 0.0)
        )
    )

    control = carla.WalkerControl()

    rgb = sensors.cameras.create_camera(
        vehicle,
        sensors.cameras.SensorTypeEnum.RGB
    )
    rgb.listen(
        lambda image: image.save_to_disk(
            'output/veh1_rgb_%06d' % image.frame_number
        )
    )
    depth = sensors.cameras.create_camera(
        vehicle,
        sensors.cameras.SensorTypeEnum.DEPTH
    )
    depth.listen(
        lambda image: image.save_to_disk(
            'output/veh1_dep_%06d' % image.frame_number,
            carla.libcarla.ColorConverter.LogarithmicDepth
        )
    )
    seg = sensors.cameras.create_camera(
        vehicle,
        sensors.cameras.SensorTypeEnum.SEGMENTATION
    )
    seg.listen(
        lambda image: image.save_to_disk(
            'output/veh1_seg_%06d' % image.frame_number,
            carla.libcarla.ColorConverter.CityScapesPalette
        )
    )

    start_time = time.time()
    timeout = 10.0  # seconds

    while time.time() <= start_time + timeout:
        control.speed = 2.0
        control.direction.x = 0.5
        control.direction.y = -0.5
        control.direction.z = 0
        ped.apply_control(control)

    rgb.destroy()
    depth.destroy()
    seg.destroy()
    vehicle.destroy()
    ped.destroy()

if __name__ == "__main__":
    main()
