import matplotlib.pyplot as plt
import numpy as np
import cv2

max_level = 0

# control variables
set_level = 5
level_setting = False
draw_quad_trees = True
show_rendering = False

# read image with matplot
img = plt.imread("my beutiful boy.jpg")


# split4 function
def split(image):
    # split the image (which is stored as an array, into two different sections) INTO A 3D ARRAYYYYY
    half_split = np.array_split(image, 2, axis=0)
    # return half_split[0] returns top half of image
    top_halves = np.array_split(half_split[0], 2, axis=1)
    bottom_halves = np.array_split(half_split[1], 2, axis=1)
    # top_halves[0] = top left half of image
    final_3D_split_array = [top_halves[0], top_halves[1], bottom_halves[0], bottom_halves[1]]
    return final_3D_split_array


def reconstruct(top_left, top_right, bottom_left, bottom_right):
    top = np.concatenate((top_left, top_right), axis=1)
    bottom = np.concatenate((bottom_left, bottom_right), axis=1)
    full = np.concatenate((top, bottom), axis=0)
    return full


def calculate_mean(image):
    return np.mean(image, axis=(0, 1))


# abstract data types can suck my willy, this one is recursive 8=====D 0:

class QuadTree:

    def input(self, img, level=0):
        self.level = level
        self.mean = calculate_mean(img).astype(int)
        self.resolution = (img.shape[0], img.shape[1])
        self.final = True

        global max_level

        if show_rendering:
            self.show_render(img)

        # record the max level that the quadtree goes to.
        if self.level >= max_level:
            max_level = self.level

        if not self.checkEqual(img, level):
            # If all pixels are not equal, or if not at final level needed, turn corner variables into the corners of corners
            # If not, algorithm stops and leaves corners
            split_img = split(img)

            self.final = False
            self.top_left = QuadTree().input(split_img[0], level + 1)
            self.top_right = QuadTree().input(split_img[1], level + 1)
            self.bottom_left = QuadTree().input(split_img[2], level + 1)
            self.bottom_right = QuadTree().input(split_img[3], level + 1)

        return self

    def show_render(self, img):
        fig, axs = plt.subplots(1, 2)

        means = np.array(list(map(lambda x: calculate_mean(x), split(img)))).astype(int).reshape(2, 2, 3)

        axs[0].imshow(img)
        axs[1].imshow(means)
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()

    def get_image(self, level):  # the level input controls the level to print
        if (self.final or self.level == level):
            mean = np.tile(self.mean, (self.resolution[0], self.resolution[1], 1))
            if draw_quad_trees:
                outline_image = cv2.rectangle(mean, (0, 0), (mean.shape[1]-1, mean.shape[0]-1), color=(225, 0, 0),
                                              thickness=1)
                return outline_image
            else:
                return mean
            # reconstructs the mean into an image by repeating the same array for the amount of resolution using numpy.tile
            # ends the recursion of getting the image as tiles are

        return reconstruct(
            # Recusion, opens the files made by input at the level needed to print
            self.top_left.get_image(level),
            self.top_right.get_image(level),
            self.bottom_left.get_image(level),
            self.bottom_right.get_image(level),
        )

    def checkEqual(self, image, level):
        first = image[0]  # gets first pixel of the image
        print(level)
        if level >= set_level and level_setting == True:  # limits the level of quadtree made
            print("hello!")
            return True
        else:
            return all((x == first).all() for x in image)  # returns true, if all pixels are equal (stops recursion) :)


quadtree = QuadTree().input(img)

# split_image = split(img)
# top_left_mean = calculate_mean(split_image[0])
# means = np.array(([calculate_mean(split_image[0]), calculate_mean(split_image[1])], [calculate_mean(split_image[2]), calculate_mean(split_image[3])]))
# means = np.array(list(map(lambda x: calculate_mean(x), split_image))).astype(int).reshape(2, 2, 3)
# print(means)
# mean_reconstruction = reconstruct(means[0,0])


# draw image plot
fig, ax = plt.subplots(figsize=(10, 10))

# show image
if level_setting:
    ax.imshow(quadtree.get_image(set_level))
else:
    ax.imshow(quadtree.get_image(max_level))

plt.show()
