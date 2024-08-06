# import cv2
# import numpy as np

# filename = "/workspace/segm.png"
# im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

# tonemapDurand = cv2.createTonemapDurand(2.2)
# ldrDurand = tonemapDurand.process(im)

# im2_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')

# new_filename = filename + "_tone_mapping.jpg"
# cv2.imwrite(new_filename, im2_8bit)


import cv2 as cv

images = []
images.append(cv.imread("/workspace/segm.png"))
merge_mertens = cv.createMergeMertens()
fusion = merge_mertens.process(images)
cv.imwrite('/workspace/segm_fusion.png', fusion * 255)