import carla

import enum


class SensorTypeEnum(enum.Enum):
    DEPTH = enum.auto()
    RGB = enum.auto()
    SEGMENTATION = enum.auto()

    @classmethod
    def print_options(self):
        for name in SensorTypeEnum.__members__.items():
            print(name)


def create_blueprint_depth(world, height, width, fov, capture_rate):
    '''
    Creates the Carla blueprint necessary for generating a depth type camera.

    Parameters
    ----------
    world: carla.World
        The Carla world in which to find the blueprint type
    height: int
        The number of vertical pixels of the camera
    width: int
        The number of horizontal pixels of the camera
    fov: int
        The Field of View (FOV) of the camera
    capture_rate: int
        The rate in which the camera captures images (Hz)

    Returns
    -------
    carla.ActorBlueprint
        A Carla actor blueprint for a depth camera sensor
    '''
    blueprint = world.get_blueprint_library().find('sensor.camera.depth')

    set_blueprint_attribute(blueprint, height, width, fov, capture_rate)
    # blueprint.set_attribute('convert', 'Depth')

    return blueprint


def create_blueprint_rgb(world, height, width, fov, capture_rate):
    '''
    Creates the Carla blueprint necessary for generating an RGB type camera.

    Parameters
    ----------
    world: carla.World
        The Carla world in which to find the blueprint type
    height: int
        The number of vertical pixels of the camera
    width: int
        The number of horizontal pixels of the camera
    fov: int
        The Field of View (FOV) of the camera
    capture_rate: int
        The rate in which the camera captures images (Hz)

    Returns
    -------
    carla.ActorBlueprint
        A Carla actor blueprint for an RGB camera sensor
    '''
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')

    set_blueprint_attribute(blueprint, height, width, fov, capture_rate)
    blueprint.set_attribute('enable_postprocess_effects', 'True')

    return blueprint


def create_blueprint_segmentation(world, height, width, fov, capture_rate):
    '''
    Creates the Carla blueprint necessary for generating an RGB type camera.

    Parameters
    ----------
    world: carla.World
        The Carla world in which to find the blueprint type
    height: int
        The number of vertical pixels of the camera
    width: int
        The number of horizontal pixels of the camera
    fov: int
        The Field of View (FOV) of the camera
    capture_rate: int
        The rate in which the camera captures images (Hz)

    Returns
    -------
    carla.ActorBlueprint
        A Carla actor blueprint for an RGB camera sensor
    '''
    blueprint = world.get_blueprint_library().find(
        'sensor.camera.semantic_segmentation'
    )

    set_blueprint_attribute(blueprint, height, width, fov, capture_rate)
    # blueprint.set_attribute('post_processing', 'SemanticSegmentation')

    return blueprint


def create_camera(
    actor,
    sensor_type=SensorTypeEnum.RGB,
    height=1080,
    width=1920,
    fov=110,
    capture_rate=5,
    transform=carla.Transform(
        carla.Location(x=0.5, y=0.0, z=1.5),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
    )
):
    '''
    Creates a Carla camera that can be used indepently or attached to an actor.

    Parameters
    ----------
    actor: carla.Actor
        The Carla actor in which to attach the camera
    sensor_type: SensorTypeEnum
        The type of sensor of the camera
    height: int
        The number of vertical pixels of the camera
    width: int
        The number of horizontal pixels of the camera
    fov: int
        The Field of View (FOV) of the camera
    capture_rate: int
        The rate in which the camera captures images (Hz)
    transform: carla.Transform
        The location in which to attach the camera relative to the actor

    Returns
    -------
    carla.Sensor
        A Carla sensor of a specific camera type
    '''
    blueprint = None
    world = actor.get_world()

    if isinstance(sensor_type, SensorTypeEnum):
        if sensor_type == SensorTypeEnum.DEPTH:
            blueprint = create_blueprint_depth(
                world,
                height,
                width,
                fov,
                capture_rate
            )
        elif sensor_type == SensorTypeEnum.RGB:
            blueprint = create_blueprint_rgb(
                world,
                height,
                width,
                fov,
                capture_rate
            )
        elif sensor_type == SensorTypeEnum.SEGMENTATION:
            blueprint = create_blueprint_segmentation(
                world,
                height,
                width,
                fov,
                capture_rate
            )
    else:
        print('The provided sensor type does not exist. Options:')
        SensorTypeEnum.print_options

    camera = world.spawn_actor(blueprint, transform, attach_to=actor)

    return camera


def set_blueprint_attribute(blueprint, height, width, fov, capture_rate):
    '''
    Sets the basic attributes used by all camera based sensors in Carla.

    Parameters
    ----------
    blueprint: carla.ActorBlueprint
        The Carla blueprint for the camera that will be modified
    height: int
        The number of vertical pixels of the camera
    width: int
        The number of horizontal pixels of the camera
    fov: int
        The Field of View (FOV) of the camera
    capture_rate: int
        The rate in which the camera captures images (Hz)
    '''
    blueprint.set_attribute('image_size_x', str(width))
    blueprint.set_attribute('image_size_y', str(height))
    blueprint.set_attribute('fov', str(fov))
    # Carla sensors want this value in seconds
    blueprint.set_attribute('sensor_tick', str(1.0/float(capture_rate)))