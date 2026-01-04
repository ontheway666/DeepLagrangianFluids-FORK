#!/usr/bin/env python3
"""This script generates random fluid sequences with SPlisHSPlasH."""
import os
import re
import argparse
from copy import deepcopy
import sys
import json
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from glob import glob
import time
import tempfile
import subprocess
from shutil import copyfile
import itertools

import open3d as o3d
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from splishsplash_config import SIMULATOR_BIN, VOLUME_SAMPLING_BIN


prm_json4mcsolver=1
prm_debug=1


fluid_velx=999
fluid_vely=999
fluid_velz=999
totalid=1

SCRIPT_DIR = os.path.dirname(__file__)

# some constants for creating the objects in the simulation
PARTICLE_RADIUS = 0.025
MAX_FLUID_START_VELOCITY_XZ = 2.0
MAX_FLUID_START_VELOCITY_Y = 0.5

MAX_RIGID_START_VELOCITY_XZ = 2.0
MAX_RIGID_START_VELOCITY_Y = 2.0

# default parameters for simulation
default_configuration = {
    "pause": False,
    "stopAt": 16.0,
    "particleRadius": 0.025,
    "numberOfStepsPerRenderUpdate": 1,
    "density0": 1000,
    "simulationMethod": 4,
    "gravitation": [0, -9.81, 0],
    "cflMethod": 0,
    "cflFactor": 1,
    "cflMaxTimeStepSize": 0.005,#prm
    "maxIterations": 100,
    "maxError": 0.01,
    "maxIterationsV": 100,
    "maxErrorV": 0.1,
    "stiffness": 50000,
    "exponent": 7,
    "velocityUpdateMethod": 0,
    "enableDivergenceSolver": True,
    "enablePartioExport": True,
    "enableRigidBodyExport": True,
    "particleFPS": 50.0,
    "partioAttributes": "density;velocity"
}

default_configuration4mc={
        "domainStart": [0.0, 0.0, 0.0],
        "domainEnd": [20, 20.0, 20.0],
        "particleRadius": 0.025,
        "numberOfStepsPerRenderUpdate": 1,
        "density0": 1000, 
        "simulationMethod": 1,
        "gravitation": [0.0, -9.81, 0.0],
        "timeStepSize": 0.004,
        "stiffness": 50000,
        "exponent": 7,
        "boundaryHandlingMethod": 0,
        "exportFrame": True,
        "exportPly":  True,
        "exportObj":  False,
        "cameraPos":[9.0, 7.5, 9.0],
        "cameraLookAt":[0.0, 4.0, 0.0],
        "cameraUp":[0.0, 1.0, 0.0],
        "pointLightPos":[11.0, 7.0, 2.0],
        "is_propeller": False,
        "solid_number": 1,
        "record_endTime": 2,
        "invisibleObjects":[0],
    }


default_simulation = {
    "contactTolerance": 0.0125,
}

default_fluid = {
    "surfaceTension": 0.2,
    "surfaceTensionMethod": 0,
    "viscosity": 0.01,
    "viscosityMethod": 3,
    "viscoMaxIter": 200,#prm
    "viscoMaxError": 0.05
}

default_fluid4mc = {
 			
}

default_rigidbody = {
    "translation": [0, 0, 0],
    "rotationAxis": [0, 1, 0],
    "rotationAngle": 0,
    "scale": [1.0, 1.0, 1.0],
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": True,
    "restitution": 0.6,
    "friction": 0.0,
    "collisionObjectType": 5,
    "collisionObjectScale": [1.0, 1.0, 1.0],
    "invertSDF": True,
}
default_rigidbody4mc = {
    "translation": [0, 0, 0],
    "rotationAxis": [0, 1, 0],
    "rotationAngle": 0,
    "scale": [1.0, 1.0, 1.0],
    "velocity": [0.0, 0.0, 0.0],
    "density": 1000.0,
    "color": [255, 255, 255], 
    "isDynamic": False,

}


default_fluidmodel = {
    "translation": [0.0, 0.0, 0.0],
     "scale": [1.0, 1.0, 1.0]}



default_fluidmodel4mc={
    "translation": [0.0, 0.0, 0.0],
     "scale": [1.0, 1.0, 1.0],
    "density": 1000.0,
    "velocity":[0,0,0],
	"color": [50, 100, 200]
}
     


def random_rotation_matrix(strength=None, dtype=None):#know
    """Generates a random rotation matrix 
    
    strength: scalar in [0,1]. 1 generates fully random rotations. 0 generates the identity. Default is 1.
    dtype: output dtype. Default is np.float32
    """
    if strength is None:
        strength = 1.0

    if dtype is None:
        dtype = np.float32

    x = np.random.rand(3)
    theta = x[0] * 2 * np.pi * strength
    phi = x[1] * 2 * np.pi
    z = x[2] * strength

    r = np.sqrt(z)
    V = np.array([np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)])

    st = np.sin(theta)
    ct = np.cos(theta)

    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    rand_R = (np.outer(V, V) - np.eye(3)).dot(Rz)
    return rand_R.astype(dtype)


def obj_volume_to_particles(objpath, scale=1, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        scale_str = '{0}'.format(scale)
        radius_str = str(radius)
        status = subprocess.run([#know
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])#zxc
        return numpy_from_bgeo(outpath)

#zxc mesh到particle并给出法向量
def obj_surface_to_particles(objpath,scalefactor=-1, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)


    if(isinstance(scalefactor,int) and scalefactor==-1):
        pass
    else:
        obj=obj.scale(scalefactor,np.array([0,0,0]))
        #都绕着物体中心放缩


    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    num_points = int(1.9 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True,\
    seed=1234)
    #zxc random

    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals


def rasterize_points(points, voxel_size, particle_radius):
    if not (voxel_size > 2 * particle_radius):
        raise ValueError(
            "voxel_size > 2*particle_radius is not true. {} > 2*{}".format(
                voxel_size, particle_radius))

    points_min = (points - particle_radius).min(axis=0)
    points_max = (points + particle_radius).max(axis=0)

    arr_min = np.floor_divide(points_min, voxel_size).astype(np.int32)
    arr_max = np.floor_divide(points_max, voxel_size).astype(np.int32) + 1

    arr_size = arr_max - arr_min

    arr = np.zeros(arr_size)

    offsets = []
    for z in range(-1, 2, 2):
        for y in range(-1, 2, 2):
            for x in range(-1, 2, 2):
                offsets.append(
                    np.array([
                        z * particle_radius, y * particle_radius,
                        x * particle_radius
                    ]))

    for offset in offsets:
        idx = np.floor_divide(points + offset, voxel_size).astype(
            np.int32) - arr_min
        arr[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return arr_min, voxel_size, arr


def find_valid_fluid_start_positions(box_rasterized, fluid_rasterized):
    """Tries to find a valid starting position using the rasterized free space and fluid"""
    fluid_shape = np.array(fluid_rasterized[2].shape)
    box_shape = np.array(box_rasterized[2].shape)
    last_pos = box_shape - fluid_shape

    valid_fluid_start_positions_arr = np.zeros(box_shape)
    for idx in itertools.product(range(0, last_pos[0] + 1),
                                 range(0, last_pos[1] + 1),
                                 range(0, last_pos[2] + 1)):
        pos = np.array(idx, np.int32)
        pos2 = pos + fluid_shape
        view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
        if np.alltrue(
                np.logical_and(view, fluid_rasterized[2]) ==
                fluid_rasterized[2]):
            if idx[1] == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
            elif np.count_nonzero(valid_fluid_start_positions_arr[idx[0],
                                                                  0:idx[1],
                                                                  idx[2]]) == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1

    valid_pos = np.stack(np.nonzero(valid_fluid_start_positions_arr), axis=-1)
    selected_pos = valid_pos[np.random.randint(0, valid_pos.shape[0])]

    # update the rasterized bounding box volume by substracting the fluid volume
    pos = selected_pos
    pos2 = pos + fluid_shape
    view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
    box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1],
                      pos[2]:pos2[2]] = np.logical_and(
                          np.logical_not(fluid_rasterized[2]), view)

    selected_pos += box_rasterized[0]
    selected_pos = selected_pos.astype(np.float) * box_rasterized[1]

    return selected_pos


def run_simulator(scene, output_dir):
    """Runs the simulator for the specified scene file"""#zxc
    with tempfile.TemporaryDirectory() as tmpdir:
        status = subprocess.run([
            SIMULATOR_BIN, '--no-cache', '--no-gui', '--no-initial-pause',
            '--output-dir', output_dir, scene
        ])


def create_fluid_data(output_dir, seed, options):

    """Creates a random scene for a specific seed and runs the simulator"""

    np.random.seed(seed)

    bounding_boxes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
                                              'Box*.obj')))
                                              #zxc 每个box对应一个不同的Obj。

    # override bounding boxes
    if options.default_box:
        bounding_boxes = [os.path.join(SCRIPT_DIR, 'models', 'Box.obj')]
    fluid_shapes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
                                            'Fluid*.obj')))
    rigid_shapes = sorted(
        glob(os.path.join(SCRIPT_DIR, 'models', 'RigidBody*.obj')))

    num_objects = np.random.choice([1, 2, 3])
    #know 1-3个水块

    # override the number of objects to generate
    if options.num_objects > 0:
        num_objects = options.num_objects
    # num_objects = random.choice([1])
    print('[num_objects]\t', num_objects)
    # create fluids and place them randomly
    def create_fluid_object():#zxc 这里差不多就是流体各种随机变换的地方
        fluid_obj = np.random.choice(fluid_shapes)
        fluid = obj_volume_to_particles(fluid_obj,
                                        scale=np.random.uniform(0.5, 1.5))[0]#zxc

        R = random_rotation_matrix(1.0)
        fluid = fluid @ R   #zxc know

        fluid_rasterized = rasterize_points(fluid, 2.01 * PARTICLE_RADIUS,
                                            PARTICLE_RADIUS)

        selected_pos = find_valid_fluid_start_positions(bb_rasterized,
                                                        fluid_rasterized)
        fluid_pos = selected_pos - fluid_rasterized[0] * fluid_rasterized[1]
        fluid += fluid_pos

        fluid_vel = np.zeros_like(fluid)
        max_vel = MAX_FLUID_START_VELOCITY_XZ
        fluid_vel[:, 0] = np.random.uniform(-max_vel, max_vel)
        fluid_vel[:, 2] = np.random.uniform(-max_vel, max_vel)
        max_vel = MAX_FLUID_START_VELOCITY_Y
        fluid_vel[:, 1] = np.random.uniform(-max_vel, max_vel)

        # global fluid_velx
        # global fluid_vely
        # global fluid_velz
        # if(prm_json4mcsolver):
        #     fluid_velx=fluid_vel[0, 0]
        #     fluid_velz=fluid_vel[0, 2]
        #     fluid_vely=fluid_vel[0, 1]


        density = np.random.uniform(500, 2000)
        viscosity = np.random.exponential(scale=1 / 20) + 0.01
        if options.uniform_viscosity:
            viscosity = np.random.uniform(0.01, 0.3)
        elif options.log10_uniform_viscosity:
            viscosity = 0.01 * 10**np.random.uniform(0.0, 1.5)

        if options.default_density:
            density = 1000
        if options.default_viscosity:
            viscosity = 0.01
        
        # print('[fluid shape]')
        # print(fluid.shape)

        # start=[np.min(fluid[:,0]),np.min(fluid[:,1]),np.min(fluid[:,2])]
        # end  =[np.max(fluid[:,0]),np.max(fluid[:,1]),np.max(fluid[:,2])]

        if(prm_json4mcsolver):
            return {
                'type': 'fluid',
                'positions': fluid,
                'velocities': fluid_vel,
                'density': density,
                'viscosity': viscosity,
            }
        else:

            return {
                'type': 'fluid',
                'positions': fluid,
                'velocities': fluid_vel,
                'density': density,
                'viscosity': viscosity,
            }

    scene_is_valid = False

    for create_scene_i in range(100):
        if scene_is_valid:
            break

        # select random bounding box
        bb_obj = np.random.choice(bounding_boxes)
        
        print('[bounding_boxes]\t'+str(len(bounding_boxes)))
        print('>>>>>>>>>>>>>>>>>>[box id]\t'+str(bb_obj))
        pattern = r'\d+'
        boxid = re.findall(pattern, str(bb_obj))[0]#know



        # convert bounding box to particles
        bb, bb_normals = obj_surface_to_particles(bb_obj)
        #zxc。box转换成粒子和normal。

        bb_vol = obj_volume_to_particles(bb_obj)[0]

        # rasterize free volume
        bb_rasterized = rasterize_points(np.concatenate([bb_vol, bb], axis=0),
                                         2.01 * PARTICLE_RADIUS,
                                         PARTICLE_RADIUS)
        bb_rasterized = bb_rasterized[0], bb_rasterized[1], binary_erosion(
            bb_rasterized[2], structure=np.ones((3, 3, 3)), iterations=3)

        objects = []

        create_fn_list = [create_fluid_object]

        for object_i in range(num_objects):

            create_fn = np.random.choice(create_fn_list)

            create_success = False
            for i in range(10):
                if create_success:
                    break
                try:
                    obj = create_fn()
    
                    objects.append(obj)
                    create_success = True
                    print('create object success')
                except:
                    print('create object failed')
                    pass

        scene_is_valid = True

        def get_total_number_of_fluid_particles():
            # if(prm_json4mcsolver):#approxmate

            #     num=0
            #     for obj in objects:
                    
            #         sz0=obj['end'][0]-obj['start'][0]
            #         sz1=obj['end'][1]-obj['start'][1]
            #         sz2=obj['end'][2]-obj['start'][2]

            #         num+=sz0*sz1*sz2/((0.025*2)**3)

            #     return num

            
            num_particles = 0
            for obj in objects:
                if obj['type'] == 'fluid':
                    num_particles += obj['positions'].shape[0]
            return num_particles

        def get_smallest_fluid_object():
            num_particles = 100000000
            obj_idx = -1
            for idx, obj in enumerate(objects):
                if obj['type'] == 'fluid':
                    # if(prm_json4mcsolver):

                    #     num=0
                    #     sz0=obj['end'][0]-obj['start'][0]
                    #     sz1=obj['end'][1]-obj['start'][1]
                    #     sz2=obj['end'][2]-obj['start'][2]

                    #     num+=sz0*sz1*sz2/((0.025*2)**3)
                    #     if(num<num_particles):
                    #         obj_idx=idx
                    #     else:
                    #         num_particles=min(num,num_particles)

                    # else:
                    if obj['positions'].shape[0] < num_particles:
                        obj_idx = idx
                    num_particles = min(obj['positions'].shape[0],
                                        num_particles)
            return obj_idx, num_particles

        total_number_of_fluid_particles = get_total_number_of_fluid_particles()
        # if(prm_json4mcsolver):
        #     if(total_number_of_fluid_particles>20000):#prm
        #         scene_is_valid=False
        # else:
        if options.const_fluid_particles:
            if options.const_fluid_particles > total_number_of_fluid_particles:
                scene_is_valid = False
            else:
                while get_total_number_of_fluid_particles(
                ) != options.const_fluid_particles:
                    difference = get_total_number_of_fluid_particles(
                    ) - options.const_fluid_particles
                    obj_idx, num_particles = get_smallest_fluid_object()
                    if num_particles < difference:
                        del objects[obj_idx]
                    else:
                        objects[obj_idx]['positions'] = objects[obj_idx][
                            'positions'][:-difference]
                        objects[obj_idx]['velocities'] = objects[obj_idx][
                            'velocities'][:-difference]

        if options.max_fluid_particles:
            if options.max_fluid_particles < total_number_of_fluid_particles:
                scene_is_valid = False

    sim_directory = os.path.join(output_dir, 'sim_{0:04d}'.format(seed))
    os.makedirs(sim_directory, exist_ok=False)
    #know

    # generate scene json file

    if(prm_json4mcsolver):#写出mcvsph能读的json
       scene = {
            'Configuration': default_configuration4mc,       
            'RigidBodies': [],
            'FluidModels': [],
        }

    else:

        scene = {
            'Configuration': default_configuration,
            'Simulation': default_simulation,
            # 'Fluid': default_fluid,
            'RigidBodies': [],
            'FluidModels': [],
        }
    rigid_body_next_id = 1

    # bounding box
    box_output_path = os.path.join(sim_directory, 'box.bgeo')
    write_bgeo_from_numpy(box_output_path, bb, bb_normals)
    #zxc

    box_obj_output_path = os.path.join(sim_directory, 'box.obj')
    copyfile(bb_obj, box_obj_output_path)#know
 
    if(prm_json4mcsolver):
        # print('[4444]')
        rigid_body = deepcopy(default_rigidbody4mc)
        rigid_body['objectId'] = 0

        rigid_body['boxid']=boxid
        rigid_body['geometryFile']= os.path.basename(
            os.path.abspath(box_obj_output_path))
        scene['RigidBodies'].append(rigid_body)#基础配置，不包含boxid

    else:
        rigid_body = deepcopy(default_rigidbody)
        rigid_body['id'] = rigid_body_next_id
        rigid_body_next_id += 1
        rigid_body['geometryFile'] = os.path.basename(
            os.path.abspath(box_obj_output_path))
        rigid_body['resolutionSDF'] = [64, 64, 64]
        rigid_body["collisionObjectType"] = 5
        scene['RigidBodies'].append(rigid_body)

    fluid_count = 0
    # print('[wewe]')
    # print(objects[0]['start'])
    # exit(0)
    for obj in objects:
        if(prm_json4mcsolver):
            fluid_id = 'fluid{0}'.format(fluid_count)
            fluid_count += 1
            fluid = deepcopy(default_fluid4mc)
            scene[fluid_id] = fluid
            
            fluid_model = deepcopy(default_fluidmodel4mc)
            global totalid
            fluid_model['objectId'] = totalid
            totalid+=1
            

            
     

            print('ndarray')
            print(type(obj['velocities']))

            fluid_model['velocity']=[
                obj['velocities'][0,0],
                obj['velocities'][0,1],
                obj['velocities'][0,2],
            ]


            fluid_output_path = os.path.join(sim_directory, fluid_id + '.bgeo')
            write_bgeo_from_numpy(fluid_output_path, obj['positions'],
                                obj['velocities'])#写出流体速度和位置信息
            fluid_model['particleFile'] = os.path.basename(fluid_output_path)
            

        else:
            fluid_id = 'fluid{0}'.format(fluid_count)
            fluid_count += 1
            fluid = deepcopy(default_fluid)
            fluid['viscosity'] = obj['viscosity']
            fluid['density0'] = obj['density']
            scene[fluid_id] = fluid

            fluid_model = deepcopy(default_fluidmodel)
            fluid_model['id'] = fluid_id

            fluid_output_path = os.path.join(sim_directory, fluid_id + '.bgeo')
            write_bgeo_from_numpy(fluid_output_path, obj['positions'],
                                obj['velocities'])#写出流体速度和位置信息
            fluid_model['particleFile'] = os.path.basename(fluid_output_path)




        scene['FluidModels'].append(fluid_model)#zxc fluid_model只有一种

    # global jsoncnt
    # if(prm_json4mcsolver):
    #     jsoncnt+=1
    #     print('[cnt+1]')
    #     scene_output_path=os.path.join('./ours_default_scenes_mc/', 'cconvscene'+str(jsoncnt)+'.json')
    # else:
    scene_output_path = os.path.join(sim_directory, 'scene.json')
    with open(scene_output_path, 'w') as f:
        json.dump(scene, f, indent=4)  
        print('[json done]\t'+str(scene_output_path))
        #know     
        #zxc 将场景特征写出到json，后续传给splish
    
    if(prm_json4mcsolver):
        pass
    else:
        if(prm_debug):
            pass
        else:
            run_simulator(os.path.abspath(scene_output_path), sim_directory)


def main():
    parser = argparse.ArgumentParser(description="Creates physics sim data")
    parser.add_argument("--output",#know
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument("--seed",
                        type=int,
                        required=True,
                        help="The random seed for initialization")
    parser.add_argument(
        "--uniform-viscosity",
        action='store_true',
        help="Generate a random viscosity value from a uniform distribution")
    parser.add_argument(
        "--log10-uniform-viscosity",
        action='store_true',
        help=
        "Generate a random viscosity value from a uniform distribution in log10 space"
    )
    parser.add_argument(
        "--default-viscosity",
        action='store_true',
        help="Force all generated fluids to have the default viscosity")
    parser.add_argument(
        "--default-density",
        action='store_true',
        help="Force all generated objects to have the default density")
    parser.add_argument(
        "--default-box",
        action='store_true',
        help="Force all generated scenes to use the default bounding box")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=0,
        help=
        "The number of objects to place in the scene. 0 (default value) means random choice from 1 to 3"
    )
    parser.add_argument(
        "--const-fluid-particles",
        type=int,
        default=0,
        help="If set a constant number of particles will be generated.")
    parser.add_argument("--max-fluid-particles",
                        type=int,
                        default=0,
                        help="If set the number of particles will be limited.")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    create_fluid_data(args.output, args.seed, args)
    #zxc 根据seed 生成场景json



if __name__ == '__main__':
    sys.exit(main())
