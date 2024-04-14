import cv2
import matplotlib.pyplot as plt
# 读取TIFF图像
image_tiff = cv2.imread(r"G:\wan\data\PVELAD_C\good_corner_tiff\img025819.tif", cv2.IMREAD_UNCHANGED)

# 检查图像是否成功读取
if image_tiff is not None:
    # 显示图像
    # cv2.imshow('TIFF Image', image_tiff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 使用matplotlib显示图像
    print('位深度:', image_tiff.dtype)
    plt.imshow(image_tiff, cmap='gray')
    plt.axis('off')
    plt.show()
else:
    print('无法读取图像文件')
