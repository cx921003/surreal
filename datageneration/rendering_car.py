import sys
import os
import random
import math
import bpy
import numpy as np
from os import getenv
from os import remove
from os.path import join, dirname, realpath, exists
from mathutils import Matrix, Vector, Quaternion, Euler
from glob import glob
from random import choice
from pickle import load
from bpy_extras.object_utils import world_to_camera_view as world2cam
import copy

sys.path.insert(0, ".")


def mkdir_safe(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


def setState0():
    for ob in bpy.data.objects.values():
        ob.select = False
    bpy.context.scene.objects.active = None


# create the different passes that we render
def create_composite_nodes(tree, params, img=None, idx=0):
    res_paths = {k: join(params['tmp_path'], k) for k in params['output_types'] if params['output_types'][k]}

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create node for foreground image
    layers = tree.nodes.new('CompositorNodeRLayers')
    layers.location = -300, 400

    # create node for background image
    bg_im = tree.nodes.new('CompositorNodeImage')
    bg_im.location = -300, 30
    if img is not None:
        bg_im.image = img

    # create node for mixing foreground and background images
    mix = tree.nodes.new('CompositorNodeMixRGB')
    mix.location = 40, 30
    mix.use_alpha = True

    # create node for the final output
    composite_out = tree.nodes.new('CompositorNodeComposite')
    composite_out.location = 240, 30

    # create node for saving depth
    if (params['output_types']['depth']):
        depth_out = tree.nodes.new('CompositorNodeOutputFile')
        depth_out.location = 40, 700
        depth_out.format.file_format = 'OPEN_EXR'
        depth_out.base_path = res_paths['depth']
        depth_out.file_slots[0].path = "%02d_" % idx

    # create node for saving normals
    if (params['output_types']['normal']):
        normal_out = tree.nodes.new('CompositorNodeOutputFile')
        normal_out.location = 40, 600
        normal_out.format.file_format = 'OPEN_EXR'
        normal_out.base_path = res_paths['normal']

    # create node for saving foreground image
    if (params['output_types']['fg']):
        fg_out = tree.nodes.new('CompositorNodeOutputFile')
        fg_out.location = 170, 600
        fg_out.format.file_format = 'PNG'
        fg_out.base_path = res_paths['fg']

    # create node for saving ground truth flow
    if (params['output_types']['gtflow']):
        gtflow_out = tree.nodes.new('CompositorNodeOutputFile')
        gtflow_out.location = 40, 500
        gtflow_out.format.file_format = 'OPEN_EXR'
        gtflow_out.base_path = res_paths['gtflow']
        gtflow_out.file_slots[0].path = "%05d_" % idx

    # merge fg and bg images
    tree.links.new(bg_im.outputs[0], mix.inputs[1])
    tree.links.new(layers.outputs['Image'], mix.inputs[2])

    tree.links.new(mix.outputs[0], composite_out.inputs[0])  # bg+fg image
    if (params['output_types']['fg']):
        tree.links.new(layers.outputs['Image'], fg_out.inputs[0])  # save fg
    if (params['output_types']['depth']):
        tree.links.new(layers.outputs['Z'], depth_out.inputs[0])  # save depth
    if (params['output_types']['normal']):
        tree.links.new(layers.outputs['Normal'], normal_out.inputs[0])  # save normal
    if (params['output_types']['gtflow']):
        tree.links.new(layers.outputs['Speed'], gtflow_out.inputs[0])  # save ground truth flow

    return (res_paths)


def create_sh_material(tree, img=None):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # uvmap = tree.nodes.new("ShaderNodeUVMap")
    #
    # uv = tree.nodes.new('ShaderNodeTexCoord')
    # uv.location = -800, 400
    #
    # uv_xform = tree.nodes.new('ShaderNodeVectorMath')
    # uv_xform.location = -600, 400
    # uv_xform.inputs[1].default_value = (0, 0, 1)
    # uv_xform.operation = 'AVERAGE'

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400
    if img is not None:
        uv_im.image = img
        tree.nodes['Image Texture'].image = img
    uv_im.select = True
    tree.nodes.active = uv_im

    rgb = tree.nodes.new('ShaderNodeRGB')
    rgb.location = -400, 200

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uvmap.outputs['UV'], uv_im.inputs['Vector'])
    tree.links.new(uv_im.outputs['Color'], emission.inputs[0])
    # tree.links.new(diffuse.outputs['BSDF'], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs['Surface'])


# computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)


def init_scene(scene, params):
    # load fbx model
    # bpy.ops.import_scene.obj( \
    #     filepath='/home/xu/data/view_synthesis/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj')

    # delete the default cube (which held the material)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete(use_global=False)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Lamp'].select = True
    bpy.ops.object.delete(use_global=False)

    # set camera properties and initial position
    bpy.ops.object.select_all(action='DESELECT')
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    scn.objects.active = cam_ob

    cam_ob.matrix_world = Matrix(((1, 0, 0, 0),
                                  (0, 0, -1, -2),
                                  (0, 1, 0, 0),
                                  (0, 0, 0, 1)))
    # cam_ob.data.angle = math.radians(40)
    cam_ob.data.lens = 60
    cam_ob.data.clip_start = 0.1
    cam_ob.data.sensor_width = 32

    # setup an empty object in the center which will be the parent of the Camera
    # this allows to easily rotate an object around the origin
    scn.cycles.film_transparent = True
    scn.render.layers["RenderLayer"].use_pass_vector = True
    scn.render.layers["RenderLayer"].use_pass_normal = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_emit = True
    scene.render.layers['RenderLayer'].use_pass_material_index = True

    # set render size
    scn.render.resolution_x = params['resy']
    scn.render.resolution_y = params['resx']
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = 'PNG'


# transformation between pose and blendshapes
def rodrigues2bshapes(pose):
    rod_rots = np.asarray(pose).reshape(24, 3)
    mat_rots = [Rodrigues(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                              for mat_rot in mat_rots[1:]])
    return (mat_rots, bshapes)


import time

start_time = None


def log_message(message):
    elapsed_time = time.time() - start_time
    print("[%.2f s] %s" % (elapsed_time, message))


def main():
    # time logging
    global start_time
    start_time = time.time()

    import argparse

    # parse commandline arguments
    log_message(sys.argv)
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--idx_cloth', type=int)

    parser.add_argument('--idx', type=int,
                        help='idx of the requested sequence')
    parser.add_argument('--ishape', type=int,
                        help='requested cut, according to the stride')
    parser.add_argument('--stride', type=int,
                        help='stride amount, default 50')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    idx = args.idx
    start_n = args.idx_cloth
    ishape = args.ishape
    stride = args.stride

    log_message("input idx: %d" % idx)
    log_message("input ishape: %d" % ishape)
    log_message("input stride: %d" % stride)

    if idx == None:
        exit(1)
    if ishape == None:
        exit(1)
    if stride == None:
        log_message("WARNING: stride not specified, using default value 50")
        stride = 50

    # import idx info (name, split)
    idx_info = load(open("pkl/idx_info.pickle", 'rb'))

    idx_info = idx_info[idx]
    idx_info['use_split'] = 'train'
    # import configuration
    log_message("Importing configuration")
    import config
    params = config.load_file('config', 'SYNTH_DATA')

    smpl_data_folder = params['smpl_data_folder']
    smpl_data_filename = params['smpl_data_filename']
    bg_path = params['bg_path']
    resy = params['resy']
    resx = params['resx']
    clothing_option = params['clothing_option']  # grey, nongrey or all
    tmp_path = params['tmp_path']
    output_path = params['output_path']
    output_types = params['output_types']
    stepsize = params['stepsize']
    clipsize = params['clipsize']
    openexr_py2_path = params['openexr_py2_path']

    # compute number of cuts
    nb_ishape = max(1, int(np.ceil((idx_info['nb_frames'] - (clipsize - stride)) / stride)))
    log_message("Max ishape: %d" % (nb_ishape - 1))
    init_scene
    if ishape == None:
        exit(1)

    assert (ishape < nb_ishape)

    # name is set given idx
    name = idx_info['name']
    tmp_path = tmp_path

    params['tmp_path'] = tmp_path

    # create tmp directory
    if not exists(tmp_path):
        mkdir_safe(tmp_path)

    # >> don't use random generator before this point <<

    # initialize RNG with seeds from sequence id
    import hashlib
    np.random.seed(start_n)

    log_message("Setup Blender")

    scene = bpy.data.scenes['Scene']
    bpy.data.materials['Material'].use_nodes = True
    scene.cycles.shading_system = True
    scene.use_nodes = True

    log_message("Listing background images")
    nh_txt_paths = []
    nh_txt_paths.append(join(bg_path, 'bg.png'))
    bg_img_name = choice(nh_txt_paths)
    bg_img = bpy.data.images.load(bg_img_name)

    res_paths = create_composite_nodes(scene.node_tree, params, img=bg_img, idx=start_n)

    log_message("Loading smpl data")

    log_message("Initializing scene")

    init_scene(scene, params)

    setState0()

    # create output directory
    if not exists(output_path):
        mkdir_safe(output_path)

    rgb_dirname = "rgb"
    rgb_path = join(tmp_path, rgb_dirname)

    # allocate
    N = 18
    cam_ob = bpy.data.objects['Camera']
    scn = bpy.context.scene
    idx = -N
    # LOOP TO RENDER
    # shapenet_dir = '/home/xu/data/view_synthesis/ShapeNetCore.v2/03001627/'  # chair

    bpy.ops.object.lamp_add(type='HEMI', radius=1000, view_align=False, location=(0, 0, 5.0))
    bpy.data.lamps['Hemi'].energy = 2.
    scene.render.use_shadows = False

    shapenet_dir =  '/home/xu/data/view_synthesis/02958343'

    with open('/home/xu/workspace/appearance-flow/data/car/train_shapes.txt', 'r') as content_file:
        model_list = sorted( [line.strip() for line in content_file] )

    stop = False
    scene.render.alpha_mode = 'TRANSPARENT'
    for model in model_list:

        print (idx / N, '/', len(model_list))

        path = os.path.join(shapenet_dir,model,'model.obj')

        idx += N
        if idx / N < start_n:
            continue
        if idx / N > start_n + 200:
            break

        if not os.path.exists(path):
            print('missing object')
            continue

        bpy.ops.import_scene.obj(filepath=path, split_mode='OFF')

        for n in range(N):
            scene.frame_set(idx + n)
            scn.objects.active = cam_ob
            angle = n * math.pi / 9.  # + math.pi   #(n) / 2 * math.pi / 2.
            cam_ob.location = (-1.7 * math.sin(angle), -1.7 * math.cos(angle), 0.9)
            # cam_ob.location = (-2 * math.sin( angle), -2 * math.cos(angle), 0.2)

            cam_ob.keyframe_insert('location', frame=idx + n)
            cam_ob.rotation_euler = (0.35 * math.pi, 0, -angle)
            # cam_ob.rotation_euler = (0.5 * math.pi, 0,  -angle)

            cam_ob.keyframe_insert('rotation_euler', frame=idx + n)
            scene.update()

        for n in range(N):
            scene.frame_set(idx + n)

            scene.render.use_antialiasing = True
            scene.render.filepath = join(rgb_path, '%05d.png' % (idx + n))
            # log_message("Rendering frame %d/%d" % (seq_frame,idx_cloth) )

            # disable render output
            logfile = '/dev/null'
            open(logfile, 'a').close()
            old = os.dup(1)
            sys.stdout.flush()
            os.close(1)
            os.open(logfile, os.O_WRONLY)

            bpy.ops.render.render(write_still=True, viewpoint=True)
            # disable output redirection
            os.close(1)
            os.dup(old)
            os.close(old)

        for obj in bpy.data.objects:
            print(obj.name)
            if 'Camera' not in obj.name and 'Lamp' not in obj.name and 'Hemi' not in obj.name:
                bpy.ops.object.select_all(action='DESELECT')
                obj.select = True
                bpy.ops.object.delete(use_global=False)
        cam_ob.animation_data_clear()


if __name__ == '__main__':
    main()

