import numpy as np


def convert_coordinates_to_raster(coords, xmax, ymin, width, height):
    w1 = 1.0 * width * width / (width + 1)
    h1 = 1.0 * height * height / (height + 1)
    xf = w1 / xmax
    yf = h1 / ymin
    coords[:, 0] *= xf
    coords[:, 1] *= yf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def process_polylist(ob, imageid, classtype, xmax, ymin, width, height):
    # todo: save poly stats (exterior and interior lengths)
    print(len(ob.geoms))

    perim_list = []
    interior_list = []

    for i, poly in enumerate(ob.geoms):
        # print(imageid, classtype, i, poly.area, poly.length)
        # print(f"[{i}]exterior length = {poly.exterior.length}")
        coords = np.array(list(poly.exterior.coords))
        coords = convert_coordinates_to_raster(coords, xmax, ymin, width, height)
        perim_list.append(coords)

        for j, poly_interior in enumerate(poly.interiors):
            # print(f"[{i}] {imageid} interior length = {poly_interior.length}")
            coords = np.array(list(poly_interior.coords))
            coords = convert_coordinates_to_raster(coords, xmax, ymin, width, height)
            interior_list.append(coords)

    return perim_list, interior_list


def process_train_sample(imageid, classtype, mpwkt, xmax, ymin):
    imgpath = os.path.join(THREE_BAND, imageid + EXT_TIFF)
    img = tiff.imread(imgpath)
    # plt.figure()
    # fig = tiff.imshow(img)
    # plt.savefig(os.path.join(DEBUG_PATH, f"{imageid}_{classtype}.png"))

    c, w, h = img.shape

    # print(f"width={w} height={h} channels={c}")

    ob = shapely.from_wkt(mpwkt) # A collection of one or more Polygons.
    exteriors, interiors = process_polylist(ob, imageid, classtype, xmax, ymin, w, h)

    # create image mask
    # print(imageid, classtype)
    img_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img_mask, exteriors, color=1)
    cv2.fillPoly(img_mask, interiors, color=0)
    # plt.imshow(img_mask)
    # plt.savefig(os.path.join(DEBUG_PATH, f"{imageid}_{classtype}_mask.png"))

    return img, img_mask


def crop_center(img, cropx, cropy):
    c, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty+cropy,startx:startx+cropx]


def crop_center_mask(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy,startx:startx+cropx]