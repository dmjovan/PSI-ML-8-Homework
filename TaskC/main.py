from copy import deepcopy
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class Sudoku:

    def __init__(self, image_path: str) -> None:

        self.image_path = os.path.join(image_path, image_path.split(os.path.sep)[-1]) + ".png"
        self.sudoku_image = None
        self.tile_width = None
        self.skip_width = None
        self.skip_width3 = None
        self.digits = {}
        self.rotation_cnts = None
        self.rot_votes = {0: 0,
                          1: 0,
                          2: 0,
                          3: 0}
        self.table = np.zeros((9,9), dtype=int)

        self.load_sudoku_image()
        self.load_digits()
        self.parse_tiles()
        self.rotate()
        
        self.table_original = deepcopy(self.table)

    def __repr__(self) -> str:
        rep = ""
        for i in range(9):
            rep += (str(list(self.table_original[i]))[1:-1]).replace(" ", "") + "\n" 
        
        for i in range(9):
            rep += (str(list(self.table[i]))[1:-1]).replace(" ", "") + ("\n" if i < 8 else '')
        
        return rep

    def load_sudoku_image(self) -> None:

        image_file = Image.open(self.image_path)
        image_file = ImageOps.grayscale(image_file)
        image = np.array(image_file)

        image[image >= 128] = 255
        image[image < 128] = 0

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

    
    def get_digit(self, num, shape = tuple) -> np.ndarray:

        image_file = self.digits[num]

        image_npy = np.array(image_file)
        image_npy = 255 - image_npy
        sum_by_col = np.sum(image_npy, axis=1)
        idxs = [i for i in range(sum_by_col.size) if sum_by_col[i] > 0]
        start_r, end_r = idxs[0], idxs[-1]
        image_npy = image_npy[start_r:(end_r+1),:]

        sum_by_col = np.sum(image_npy, axis=0)
        idxs = [i for i in range(sum_by_col.size) if sum_by_col[i] > 0]
        start_c, end_c = idxs[0], idxs[-1]
        image_npy = image_npy[:, start_c:(end_c+1)]


        image_file = Image.fromarray(image_npy)

        #s = time.time()
        image_file = image_file.resize(shape)
        #print(time.time() - s)
        digit = np.array(image_file)

        digit[digit >= 128] = 255
        digit[digit < 128] = 0

        return digit

    def load_digits(self) -> None:

        dir_name = os.path.dirname(self.image_path)
        for i in range(1,10):
            image_file = Image.open(os.path.join(dir_name, "digits", str(i) + ".png"))
            image = np.array(image_file)
            image = 255 - image[:, :, 3]

            img = Image.fromarray(image, 'L')
            image_file = ImageOps.grayscale(img)
            image_file = image_file.resize((self.tile_width, self.tile_width))

            self.digits[i] = image_file


    def parse_tiles(self):

        self.tile_number_dict = {}

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

            if isinstance(tile_number, int):
                self.add(tile_number, (row, col))
            else:
                self.tile_number_dict[i] = tile_number

        
    def get_number_from_tile(self, tile: np.ndarray) -> int:

        def test_number(tile: np.ndarray, num: int, k: int=0) -> int:
            if k == 0:
                # crop only digit
                sum_by_col = np.sum(tile, axis=1)
                idxs = [i for i in range(sum_by_col.size) if sum_by_col[i] > 0]
                start_r, end_r = idxs[0], idxs[-1]
                tile = tile[start_r:(end_r+1),:]

                sum_by_col = np.sum(tile, axis=0)
                idxs = [i for i in range(sum_by_col.size) if sum_by_col[i] > 0]
                start_c, end_c = idxs[0], idxs[-1]
                tile = tile[:, start_c:(end_c+1)]

                H,W = tile.shape
                
                #s = time.time()
                digits_transformed = self.get_digit(num, (W,H))
                #print(time.time() - s)
                
                plt.figure()
                plt.imshow(tile)
                plt.show()
                plt.figure()
                plt.imshow(digits_transformed)
                plt.show()

                
                digit_product = (tile  == digits_transformed).sum()
                #print(time.time() - s)
      
            else:

                # crop only digit
                sum_by_col = np.sum(tile, axis=1)
                idxs = [i for i in range(sum_by_col.size) if sum_by_col[i] > 0]
                start_r, end_r = idxs[0], idxs[-1]
                tile = tile[start_r:(end_r+1),:]

                sum_by_col = np.sum(tile, axis=0)
                idxs = [i for i in range(sum_by_col.size) if sum_by_col[i] > 0]
                start_c, end_c = idxs[0], idxs[-1]
                tile = tile[:, start_c:(end_c+1)]

                tile_transformed = np.rot90(tile, k=k, axes=(0,1))
                H,W = tile_transformed.shape

                digits_transformed = self.get_digit(num, (W,H))

                plt.figure()
                plt.imshow(tile_transformed)
                plt.show()
                plt.figure()
                plt.imshow(digits_transformed)
                plt.show()

                digit_product = (tile_transformed  == digits_transformed).sum()
            print(digit_product)

            return digit_product


        if np.sum(tile) / 255 == tile.size : # empty tile
            return 0

        tile = 255 - tile

        number_per_rot = {}

        if self.rotation_cnts is None:
            test_rotation = np.zeros((4,9), dtype=int)
            
            for k in range(4):
                test_rotation[k, :] = np.array([test_number(tile, num, k) for num in range(1,10)])
                number_per_rot[k] = np.argmax(test_rotation[k, :]) + 1
                
            # k_opt1 = np.argwhere(test_rotation - )

            k_opt, number = np.unravel_index(np.argmax(test_rotation, axis=None), test_rotation.shape)

            # self.rotation_cnts = k_opt
            self.rot_votes[k_opt] += 1

            # number += 1


        return number_per_rot

    def rotate(self) -> None:

        if self.rotation_cnts is None:
            self.rotation_cnts = max(self.rot_votes, key=self.rot_votes.get)
            if self.rot_votes[1] + self.rot_votes[3] > 0:
                if self.rot_votes[1] > self.rot_votes[3]:
                    self.rotation_cnts = 1
                else:
                    self.rotation_cnts = 3
            print(self.rotation_cnts)
                    
            for i in range(81):
                if i in self.tile_number_dict.keys():
                    row = i // 9
                    col = i % 9
                    # print(row, col, self.tile_number_dict[i])
                    self.add(self.tile_number_dict[i][self.rotation_cnts], (row, col))

        if self.rotation_cnts > 0:
            self.table = np.rot90(self.table, k=4-self.rotation_cnts, axes=(0,1))

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

    # path = input().strip("\n")
    for t in range(5, 6):
        print('=========================================')
        tt = "0" + str(t)
        path = f"/home/grbic/Desktop/PSI-ML-8-Homework/TaskC/dataset/public/set/{tt}"
        #import time
        #start = time.time()
        sudoku = Sudoku(path)
        sudoku.solve()
        print(sudoku)

        with open(f"/home/grbic/Desktop/PSI-ML-8-Homework/TaskC/dataset/public/outputs/{tt}.txt", 'r') as f:
            expected = ('').join(f.readlines())

        print('------------------------------')
        print(expected)