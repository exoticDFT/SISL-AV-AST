import sensors.cameras

import carla

import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(3.0)

    world = client.get_world()

    view_trans = carla.Transform(
        carla.Location(0.0, 0.0, 100.0),
        carla.Rotation(90.0, 0.0, 0.0)
    )

    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    sensors.cameras.set_blueprint_attribute(cam_bp, 480, 480, 120, 60.0)
    # cam_bp.set_attribute('image_size_x', '480')
    # cam_bp.set_attribute('image_size_y', '480')
    # cam_bp.set_attribute('fov', '120')
    # cam_bp.set_attribute('sensor_tick', str(1.0/60.0))
    camera = world.try_spawn_actor(cam_bp, view_trans)

    # camera = sensors.cameras.create_camera(world, transform=view_trans)

    # Timing variables
    timeout = 60.0 * 0.2 # seconds * minutes
    start_time = time.time()

    # Start recording
    outfile = "Test01.log"
    print("Recording on file: %s" % client.start_recorder(outfile))
    
    # Do things during recording
    while time.time() < start_time + timeout:
        pass

    # Stop recording
    client.stop_recorder()

    # Replay the video
    # client.replay_file(outfile, 0.0, 100.0, camera.)

    # Destroy the camera
    camera.destroy()


if __name__ == "__main__":
    main()