import os
import time
import numpy as np
from PIL import Image, ImageOps

class Sudoku:

    def __init__(self, image_path: str) -> None:

        self.image_path = image_path
        self.sudoku_image = None
        self.tile_width = None
        self.skip_width = None
        self.skip_width3 = None
        self.digits = {}
        self.rotation_cnts = None
        self.table = np.zeros((9,9), dtype=int)

        self.load_sudoku_image()
        self.load_digits()
        self.parse_tiles()
        self.rotate()

    def __repr__(self) -> str:
        rep = ""
        for i in range(9):
            rep += str(list(self.table[i]))[1:-1] + "\n"

        return rep

    def load_sudoku_image(self) -> None:

        image_file = Image.open(self.image_path)
        image_file = ImageOps.grayscale(image_file)
        image = np.array(image_file)

        image[image > 0] = 255

        height, width = image.shape

        cols = np.sum(image, axis=0)
        rows = np.sum(image, axis=1)
        
        upper_left_x = np.argmax(rows > 0)
        upper_left_y = np.argmax(cols > 0)

        lower_right_x = height - 1 - np.argmax(np.flip(rows) > 0)
        lower_right_y = width - 1 - np.argmax(np.flip(cols) > 0)

        self.sudoku_image = image[upper_left_x:lower_right_x, upper_left_y:lower_right_y]

        p = self.sudoku_image.shape[1] // 9
        o = self.sudoku_image.shape[1] % 9

        if o > 4:
            p += 1

        self.tile_width = int(np.sum(self.sudoku_image[0,:p])/255)

        t = int(1.5 * p)
        self.skip_width = int(t - np.sum(self.sudoku_image[0,:t])/255)
        self.skip_width3 = int((self.sudoku_image.shape[1] - 6*self.skip_width - 9*self.tile_width + 1)/2)

    def load_digits(self) -> None:

        dir_name = os.path.dirname(self.image_path)
        for i in range(1,10):
            image_file = Image.open(dir_name + "/digits/" + str(i) + ".png")
            image_resized = image_file.resize((self.tile_width, self.tile_width))
            image = np.array(image_resized)
            image = 255 - image[:, :, 3]

            img = Image.fromarray(image, 'L')
            image_file = ImageOps.grayscale(img)
            digit_img = np.array(image_file)

            digit = 255 - digit_img
            digit[digit > 0] = 255

            self.digits[i] = digit

    def parse_tiles(self):

        for i in range(81):
            row = i // 9
            col = i % 9

            num_skip_width3_row = row // 3
            num_skip_width_row = row - num_skip_width3_row

            num_skip_width3_col = col // 3
            num_skip_width_col = col - num_skip_width3_col

            vertical_start = self.tile_width * row + self.skip_width * num_skip_width_row + self.skip_width3 * num_skip_width3_row
            horizontal_start = self.tile_width * col + self.skip_width * num_skip_width_col + self.skip_width3 * num_skip_width3_col

            tile = self.sudoku_image[vertical_start: vertical_start + self.tile_width, horizontal_start: horizontal_start + self.tile_width]
            tile_number = self.get_number_from_tile(tile)
            
            self.add(tile_number, (row, col))

        
    def get_number_from_tile(self, tile: np.ndarray) -> int:

        H,W = tile.shape

        def test_number(tile: np.ndarray, num: int, k: int=0) -> int:
            if k == 0:
                digit_product = tile * self.digits[num][0:H, 0:W]
            else:
                digit_product = tile * np.rot90(self.digits[num][0:H, 0:W], k=k, axes=(0,1)).reshape((H,W))

            digit_product[digit_product > 0] = 255

            return np.sum(digit_product) / 255


        if np.sum(tile) / 255 == tile.size : # empty tile
            return 0

        tile = 255 - tile

        if self.rotation_cnts is None:
            lst  = []
            for k in range(4):
                lst.append([test_number(tile, num, k) for num in range(1,10)])
            
            test_rotation = np.array(lst)
            k_opt, number = np.unravel_index(np.argmax(test_rotation, axis=None), test_rotation.shape)

            self.rotation_cnts = k_opt

            number += 1

        else:
            white_pixels = np.array([test_number(tile, num, self.rotation_cnts) for num in range(1,10)])
            number = np.argmax(white_pixels) + 1

        return number

    def rotate(self) -> None:

        print('rotation', self.rotation_cnts)

        if self.rotation_cnts > 0:
            self.table = np.rot90(self.table, k=self.rotation_cnts, axes=(0,1))

    def add(self, number: int, position: tuple) -> None:
        self.table[position] = number

    def is_valid(self, number: int, position: tuple) -> bool:

        row, col = position

        # check row and column
        if (number in list(self.table[row, :])) or (number in list(self.table[:, col])):
            return False

        # box check
        box_x = position[0] // 3
        box_y = position[1] // 3

        if number in list((self.table[box_x*3: box_x*3+3, box_y*3: box_y*3+3]).flatten()):
            return False

        return True

    def find_empty(self):
        
        empty = np.argwhere(self.table == 0)
        empty = [tuple(e) for e in empty]

        return empty.pop(0) if empty else empty

    def solve(self) -> bool:

        pos = self.find_empty()

        if not pos:
            return True

        for num in range(1,10):
            if self.is_valid(num, pos):
                self.add(num, pos)

                if self.solve():
                    return True
                
                self.add(0, pos)

        return False


if __name__ == "__main__":

    # path = input()

    for i in range(10):

        path = r"TaskC\dataset\public\set\0" + str(i) + r"\0" + str(i) + ".png"
        print(path)

        start = time.time()

        sudoku = Sudoku(path)

        print(sudoku)
        # sudoku.solve()
        # print(sudoku)
        
        print(time.time() - start)
        print('-------------------------------------------')

        del sudoku
