import cv2
import os
import click
import numpy as np
import copy
import OpenEXR, Imath, Image
import random
import pandas as pd
@click.command()
@click.option("--type", type=str,
              default=None, help="Input image.")
@click.option("--input", type=click.Path(exists=True, dir_okay=True),
              default=None, help="Input image.")
@click.option("--output", type=click.Path(exists=True, dir_okay=True),
              default=None, help="Output dest")
def convert(type,input,output):
    if type == 'depth':
        return convert_depth(input,output)
    elif type == 'flow':
        return convert_flow(input, output)
def convert_flow(input, output):
    count = 0
    for root, folder, files in os.walk(input):
        for file in sorted(files):
            suffix = os.path.basename(file).split('.')[-1]
            filename = os.path.basename(file).split('.')[0]
            if suffix != 'exr':
                continue
            image = OpenEXR.InputFile(os.path.join(input, file))

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            dw = image.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            fx_str = image.channel('R', pt)
            fx = Image.frombytes("F", size, fx_str)
            fx = np.array(fx)/ size[0] * 2
            fx = np.reshape(fx,(fx.shape[0], fx.shape[1],1))
            fy_str = image.channel('G', pt)
            fy = Image.frombytes("F", size, fy_str)
            fy = np.array(fy)/ size[1] * 2
            fy = np.reshape(fy,(fy.shape[0], fy.shape[1],1))

            flow = np.concatenate((fx,fy),axis=2)
            # print count,np.max(fx)
            #
            # fx = cv2.convertScaleAbs(fx*255)
            # print count,np.max(fx)
            #
            # fx = fx.astype(np.uint8)
            # cv2.imshow("s",fx)
            # cv2.waitKey()
            # count+=1

            np.save(os.path.join(output,filename+".npy"),flow)

def convert_depth(input, output):
    count = 0
    for root, folder, files in os.walk(input):
        for file in sorted(files):
            suffix = os.path.basename(file).split('.')[-1]
            filename = os.path.basename(file).split('.')[0]
            if suffix != 'exr':
                continue
            image = OpenEXR.InputFile(os.path.join(input, file))

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            dw = image.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            depth_str = image.channel('R', pt)
            depth = np.fromstring(depth_str, dtype=np.float32)
            depth.shape = (size[1], size[0])

            depth[depth > 100] = 0.
            # depth -= 4.
            # depth = (depth + 1) / (2.)
            # foc = 60 / 1000.
            # depth = (np.array(depth) - 4) / foc
            # print set(depth.flatten())
            # depth[depth>256] = 0
            # depth = np.reshape(depth,(depth.shape[0], depth.shape[1],1))

            # cv2.imshow("s",depth.astype(np.uint8))
            # cv2.waitKey()

            np.save(os.path.join(output,filename+".npy"),depth.astype(np.float32))



if __name__ == '__main__':
    #segments2sketches()
    # simplify_images()
    convert()
