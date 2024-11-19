# https://github.com/dakshaau/map_tile_download
# http://www.f-bmpl.com/index.php/faites-le-vous-meme/210-3-listing-des-serveurs-ortos
####################################################################################################################
# VirtualEarth - Textures IGN pour la France                                                                       #
#                                                                                                                  #
# Examples                                                                                                         #
#                                                                                                                  #
# Map (png)                                                                                                        #
# http://r3.ortho.tiles.virtualearth.net/tiles/r0230102122203313.png?g=45                                          #
# Satellite (jpg)                                                                                                  #
# http://a0.ortho.tiles.virtualearth.net/tiles/a023010212220331013.jpeg?g=45                                       #
# Hybrid (jpg)                                                                                                     #
# http://h1.ortho.tiles.virtualearth.net/tiles/h0230102122203031121.jpeg?g=45                                      #
#                                                                                                                  #
#                                                                                                                  #
# Procédures :                                                                                                     #
#                                                                                                                  #
# http://[type][server].ortho.tiles.virtualearth.net/tiles/[type][location].[format]?g=45                          #
#                                                                                                                  #
# * Servers: Choice of four, encountered: 0, 1, 2, 3.                                                              #
# * Types: Note that this appears twice in the URL. r = road, a = aerial, h = hybrid.                              #
# * Formats: png for road, jpeg otherwise.                                                                         #
# * Location: Array of successive zooms, starting at empty string for whole planet.                                #
# o 0 = upper left, 1 = upper right, 2 = lower left, 3 = lower right.                                              #
# o Number of digits = amount of zoom, 0 = whole planet (only 1 and higher, northern hemisphere are available),    #
#                                      19 = maximum zoom for certain metro areas.                                  #
####################################################################################################################

import os
import cv2
import requests
import numpy as np
from maps.bing_maps.bing import TileSystem


def download_tiles(lt_lat, lt_lng, rb_lat, rb_lng, style='a'):
    '''
    For the lat long coordinated to be positined as Top Left and Bottom Right
    the topleft_lat > bottomright_lat and topleft_long < bottomright_long

    If the above condition doesn't hold then the coodinated need to be swaped
    accordingly
    '''

    if style is None:
        style = 'a'

    if style == 'r' or style == 'h' or style == 'a':
        pass
    else:
        style = 'a'

    if style == 'r':
        tile_net_path = 'http://r0.ortho.tiles.virtualearth.net/tiles/r'
    elif style == 'a':
        tile_net_path = 'http://a0.ortho.tiles.virtualearth.net/tiles/a'
    elif style == 'h':
        tile_net_path = 'http://h0.ortho.tiles.virtualearth.net/tiles/h'
    else:
        tile_net_path = 'http://a0.ortho.tiles.virtualearth.net/tiles/a'

    if lt_lat == rb_lat or lt_lng == rb_lng:
        print('Cannot accept equal latitude or longitude pairs.\nTry with a different combination')
        exit(0)

    if lt_lat > rb_lat and lt_lng > rb_lng:
        temp = lt_lng
        lt_lng = rb_lng
        rb_lng = temp
    if lt_lat < rb_lat and lt_lng < rb_lng:
        temp = lt_lat
        lt_lat = rb_lat
        rb_lat = temp
    elif lt_lat < rb_lat and lt_lng > rb_lng:
        temp = lt_lng
        lt_lng = rb_lng
        rb_lng = temp
        temp = lt_lat
        lt_lat = rb_lat
        rb_lat = temp

    lb_lat = lt_lat
    lb_lng = rb_lng
    rt_lat = rb_lat
    rt_lng = lt_lng
    bnd_sqr = [(lt_lat, lt_lng), (rt_lat, rt_lng), (lb_lat, lb_lng), (rb_lat, rb_lng)]
    # print(bnd_sqr)
    t = TileSystem()
    # print(t.EarthRadius)
    emptyImage = cv2.imread('maps/bing_maps/Error.jpeg', 0)

    ## http://a0.ortho.tiles.virtualearth.net/tiles/a120200223.jpeg?g=2
    # qKey = '1202002230022122121212'
    levels = []
    keys = []
    # l_v = np.arange(lt_lng, lb_lng, 0.00001).tolist()
    # l_v = [(lt_lat, l) for l in l_v]
    # prevkey = ''
    '''
    Downloading the maximum levelOfDetail Map Tile available for the four corners of the bounding
    rectangle.

    '''
    if not os.path.exists('maps/Images'):
        os.mkdir('maps/Images')

    _, __, files = list(os.walk('maps/Images'))[0]
    for file in files:
        os.remove(os.path.join(_, file))

    for i, (lat, lng) in enumerate(bnd_sqr):
        detail = 23
        # tx, ty = t.QuadKeyToTileXY(qKey)
        # px, py = t.TileXYToPixelXY(tx, ty)
        # lat, lng = t.PixelXYToLatLong(px, py, detail)
        px, py = t.LatLongToPixelXY(lat, lng, detail)
        tx, ty = t.PixelXYToTileXY(px, py)
        qKey = t.TileXYToQuadKey(tx, ty, detail)
        empty = 0
        while empty == 0:
            fileName = str(i)
            file = open('maps/Images/seq_{}.jpeg'.format(fileName), 'wb')
            response = requests.get(tile_net_path + '{}.jpeg?g=2'.format(qKey),
                                    stream=True)

            if not response.ok:
                # Something went wrong
                print('Invalid depth')

            for block in response.iter_content(1024):
                file.write(block)
            file.close()
            curimage = cv2.imread('maps/Images/seq_{}.jpeg'.format(fileName), 0)
            # while True:
            # 	key = cv2.waitKey(10)
            # 	if key == 27:
            # 		break
            # 	cv2.imshow('disp',curimage - emptyImage)
            empty = np.where(np.not_equal(curimage, emptyImage))[0].shape[0]
            # print(empty)
            if empty == 0:
                detail -= 1
                px, py = t.LatLongToPixelXY(lat, lng, detail)
                tx, ty = t.PixelXYToTileXY(px, py)
                qKey = t.TileXYToQuadKey(tx, ty, detail)
            # print('Moving on, new QuadKey : {}'.format(qKey))
        levels.append(detail)
        keys.append(qKey)
    min_level = min(levels)
    pixelXY = []

    [os.remove('maps/Images/seq_{}.jpeg'.format(i)) for i in range(4)]
    # keys = []
    # print(levels)
    print('Selected levelOfDetail: {}'.format(min_level))
    '''
    Finding out the maximum common levelOfDetail for the tiles and redownloading accordingly.

    '''
    tileXY = []
    tilePixelXY = []
    for i, (level, (lat, lng)) in enumerate(zip(levels, bnd_sqr)):
        # print(level)
        # if level > min_level:
        # print('Blah')
        # lat, lng = coors
        px, py = t.LatLongToPixelXY(lat, lng, min_level)
        pixelXY.append((px, py))
        tx, ty = t.PixelXYToTileXY(px, py)
        tileXY.append((tx, ty))
        tpx, tpy = t.PixelXYToTilePixelXY(px, py)
        tilePixelXY.append((tpx, tpy))
        qKey = t.TileXYToQuadKey(tx, ty, min_level)
        fileName = '{},{}'.format(tx, ty)
        file = open('maps/Images/seq_{}.jpeg'.format(fileName), 'wb')
        response = requests.get(tile_net_path + '{}.jpeg?g=2'.format(qKey), stream=True)

        if not response.ok:
            # Something went wrong
            print('Invalid depth')

        for block in response.iter_content(1024):
            file.write(block)
        file.close()
        # keys.append(qKey)
        keys[i] = qKey
    # print(keys)
    # print(pixelXY)
    # print(tilePixelXY)
    print('Downlaoded corner tiles.')

    '''
    Calculating the pixelXY for lat long with
    '''
    tb = pixelXY[2][0] - pixelXY[0][0] + 1
    lr = pixelXY[1][1] - pixelXY[0][1] + 1
    # print(tb, lr)

    # print(tilePixelXY[0], tilePixelXY[2])
    # print(tilePixelXY[1], tilePixelXY[3])
    tileD_tb = (256 - tilePixelXY[0][0]) + tilePixelXY[2][0] + 1
    tileD_lr = (256 - tilePixelXY[0][1]) + tilePixelXY[1][1] + 1

    if (tileXY[1][1] - tileXY[0][1]) > 1 and (tileXY[2][0] - tileXY[0][0]) > 1:
        tb -= tileD_tb
        lr -= tileD_lr
    elif (tileXY[1][1] - tileXY[0][1]) > 1:
        lr -= tileD_lr
        tb = 0
    elif (tileXY[2][0] - tileXY[0][0]) > 1:
        tb -= tileD_tb
        lr = 0
    else:
        tb = 0
        lr = 0

    # print(tb/256, lr/256)
    if tb > 20000 or lr > 20000:
        print(int(tb / 256), int(lr / 256))
        print('Too many tiles. Reduce the bounding rectangle area!')
        exit(0)

    num_tiles_lr = int(lr / 256)
    num_tiles_tb = int(tb / 256)
    # print(num_tiles_tb, num_tiles_lr)

    if num_tiles_tb > 0 and num_tiles_lr > 0:
        prog = 0.
        tot = (num_tiles_lr + 2) * (num_tiles_tb + 2)
        count = 0.
        print('Downloading remaining tiles, {} ...'.format(tot))
        for i in range(0, num_tiles_tb + 2):
            tx = tileXY[0][0] + i
            for j in range(0, num_tiles_lr + 2):
                ty = tileXY[0][1] + j
                qKey = t.TileXYToQuadKey(tx, ty, min_level)
                fileName = '{},{}'.format(tx, ty)
                file = open('maps/Images/seq_{}.jpeg'.format(fileName), 'wb')
                response = requests.get(tile_net_path + '{}.jpeg?g=2'.format(qKey),
                                        stream=True)

                if not response.ok:
                    # Something went wrong
                    print('Invalid depth')

                for block in response.iter_content(1024):
                    file.write(block)
                count += 1.
                prog = (count / tot) * 100
                print('\rCompleted: {:.2f}%'.format(prog), end=' ')
        print()

    elif num_tiles_tb > 0:
        prog = 0.
        tot = num_tiles_tb
        count = 0.
        print('Downloading remaining tiles, {} ...'.format(tot))
        for i in range(1, num_tiles_tb + 1):
            tx = tileXY[0][0] + i
            ty = tileXY[0][1]
            qKey = t.TileXYToQuadKey(tx, ty, min_level)
            fileName = '{},{}'.format(tx, ty)
            file = open('maps/Images/seq_{}.jpeg'.format(fileName), 'wb')
            response = requests.get(tile_net_path + '{}.jpeg?g=2'.format(qKey),
                                    stream=True)

            if not response.ok:
                # Something went wrong
                print('Invalid depth')

            for block in response.iter_content(1024):
                file.write(block)
            count += 1.
            prog = (count / tot) * 100
            print('\rCompleted: {:.2f}%'.format(prog), end=' ')
        print()

    elif num_tiles_lr > 0:
        prog = 0.
        tot = num_tiles_lr
        count = 0.
        print('Downloading remaining tiles, {} ...'.format(tot))
        for i in range(1, num_tiles_lr + 1):
            tx = tileXY[0][0]
            ty = tileXY[0][1] + i
            qKey = t.TileXYToQuadKey(tx, ty, min_level)
            fileName = '{},{}'.format(tx, ty)
            file = open('maps/Images/seq_{}.jpeg'.format(fileName), 'wb')
            response = requests.get(tile_net_path + '{}.jpeg?g=2'.format(qKey),
                                    stream=True)

            if not response.ok:
                # Something went wrong
                print('Invalid depth')

            for block in response.iter_content(1024):
                file.write(block)
            file.close()
            count += 1.
            prog = (count / tot) * 100
            print('\rCompleted: {:.2f}%'.format(prog), end=' ')
        print()

    file = open('maps/bing_maps/params.dat', 'w')
    file.write(
        '{} {} {} {}'.format(tilePixelXY[0][0], 256 - tilePixelXY[2][0], tilePixelXY[0][1], 256 - tilePixelXY[1][1]))
    file.close()


def stitching_tiles(map_name: str):
    print('\nStitching together images ...')

    _, __, files = list(os.walk('maps/Images'))[0]
    # print(len(files))
    files.sort(key=lambda x: (int(x.split(',')[0].split('_')[-1]), int(x.split(',')[1].split('.')[0])))
    # print(files[0])
    tilX = lambda x: int(x.split(',')[0].split('_')[-1])
    tilY = lambda x: int(x.split(',')[1].split('.')[0])
    Xs = sorted(list(set([tilX(x) for x in files])))
    Ys = sorted(list(set([tilY(x) for x in files])))
    fin_img = None
    vertical = None
    prev_x = None
    count = 0.
    tot = len(Xs) * len(Ys)
    prog = 0.
    for x in Xs:
        vertical = None
        for y in Ys:
            img = cv2.imread('maps/Images/seq_{},{}.jpeg'.format(x, y))
            if vertical is None:
                vertical = img
            else:
                vertical = np.concatenate((vertical, img), axis=0)
            count += 1.
            prog = count / tot * 100
            print('\rCompleted: {:.2f}%'.format(prog), end=' ')
        if fin_img is None:
            fin_img = vertical
        else:
            fin_img = np.concatenate((fin_img, vertical), axis=1)

    print()
    print()
    # cv2.imwrite('ArielView_org.jpeg', fin_img)
    h, w = fin_img.shape[:2]
    re_img = None
    if h <= w:
        ratio = float(h / w)
        re_img = cv2.resize(fin_img, (720, int(ratio * 720)))
    else:
        ratio = float(w / h)
        re_img = cv2.resize(fin_img, (int(ratio * 720), 720))
    # while True:
    #     key = cv2.waitKey(10)
    #     if key == 27:
    #         break
    #     cv2.imshow('StitchedImage', re_img)

    f = open('maps/bing_maps/params.dat', 'r')
    params = f.readlines()[0]
    params.strip()
    tc, bc, tr, br = [int(x) for x in params.split(' ')]
    fin_img = fin_img[tr:-br, tc:-bc, :]
    path = "maps/" + map_name + '/ArielView.jpeg'
    cv2.imwrite(path, fin_img)
    print('Saved image at: {}'.format(os.path.abspath(path)))
