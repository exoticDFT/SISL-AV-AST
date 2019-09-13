#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import util.client
import sensors.cameras


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "Quit"
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return "Quit"
            elif event.key == pygame.K_r:
                return "Record"
            elif event.key == pygame.K_p:
                return "Auto"


def generate_scenario(world):
    dynamic = []
    static = []

    for actor in static:
        actor.set_simulate_physics(False)


def main():
    actor_list = []
    pygame.init()

    is_recording = False
    autopilot = False
    images = []

    display = pygame.display.set_mode(
        (1600, 900),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = util.client.create('localhost', 2000, 2.0)

    world = client.load_world('Town05')

    weather = carla.WeatherParameters(
            cloudyness=0.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            wind_intensity=0.0,
            sun_azimuth_angle=130.0,
            sun_altitude_angle=68.0)
    world.set_weather(weather)

    try:
        m = world.get_map()
        start_pose = m.get_spawn_points()[152]
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        actor_list.append(vehicle)

        bb = vehicle.bounding_box.extent

        cam_transform = carla.Transform(
            carla.Location(x=bb.x, z=bb.z),
            carla.Rotation(pitch=0.0)
        )

        camera_rgb = sensors.cameras.create_camera(
            vehicle,
            sensor_type=sensors.cameras.SensorTypeEnum.RGB,
            height=900,
            width=1600,
            fov=70,
            capture_rate=12,
            transform=cam_transform
        )
        actor_list.append(camera_rgb)

        camera_depth = sensors.cameras.create_camera(
            vehicle,
            sensor_type=sensors.cameras.SensorTypeEnum.DEPTH,
            height=900,
            width=1600,
            fov=70,
            capture_rate=12,
            transform=cam_transform
        )
        actor_list.append(camera_depth)

        camera_semseg = sensors.cameras.create_camera(
            vehicle,
            sensor_type=sensors.cameras.SensorTypeEnum.SEGMENTATION,
            height=900,
            width=1600,
            fov=70,
            capture_rate=12,
            transform=cam_transform
        )
        actor_list.append(camera_semseg)

        # Create a synchronous mode context.
        with CarlaSyncMode(
            world,
            camera_rgb,
            camera_depth,
            camera_semseg,
            fps=12
        ) as sync_mode:
            while True:
                event = handle_events()

                if event == "Quit":
                    return
                elif event == "Record":
                    is_recording = not is_recording
                elif event == "Auto":
                    autopilot = not autopilot
                    vehicle.set_autopilot(autopilot)

                clock.tick()

                # Advance the simulation and wait for the data.
                (
                    snapshot,
                    image_rgb,
                    image_depth,
                    image_semseg
                ) = sync_mode.tick(timeout=2.0)

                image_depth.convert(carla.ColorConverter.LogarithmicDepth)
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_depth, blend=True)
                draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render(
                        '% 5d FPS (real)' % clock.get_fps(),
                        True,
                        (255, 255, 255)
                    ),
                    (8, 10)
                )
                display.blit(
                    font.render(
                        '% 5d FPS (simulated)' % fps,
                        True,
                        (255, 255, 255)
                    ),
                    (8, 28)
                )
                location = vehicle.get_location()
                display.blit(
                    font.render(
                        'Location: ({}, {})'.format(location.x, location.y),
                        True,
                        (255, 255, 255)
                    ),
                    (8, 46)
                )
                pygame.display.flip()

                if is_recording:
                    images.append(
                        (
                            image_rgb,
                            image_depth,
                            image_semseg
                        )
                    )

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        if images:
            print('Writing images to disk (this may take a while.)')

            for image in images:
                image[0].save_to_disk('_out/%08d-rgb' % image[0].frame)
                image[1].save_to_disk('_out/%08d-dep' % image[1].frame)
                image[2].save_to_disk('_out/%08d-seg' % image[2].frame)

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
