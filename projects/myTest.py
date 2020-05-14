import os
os.chdir('../')
cur_path = os.getcwd()

if __name__ == '__main__':
    # from tests.test_pipelines.test_loading import TestLoading
    # complete_test = TestLoading()
    # complete_test.setup_class()
    # complete_test.test_load_multi_channel_img()
    # print("Testing finished!")
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread("D:\Code\python\mmdetection\\tests\data\color.jpg", )
    plt.imshow(img)
    plt.show()
    print("Testing finished!")