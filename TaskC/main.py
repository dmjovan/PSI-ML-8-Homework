import numpy as np
from PIL import Image, ImageOps

class Sudoku:

    def __init__(self, image_path = r"sudoku-public\public\set\00\00.png") -> None:

        self.image_path = image_path
        self.sudoku_image = None
        self.tile_width = None
        self.skip_width = None
        self.skip_width3 = None
        self.table = np.zeros((9,9), dtype=int)

        self.load_sudoku_image()

    def __repr__(self) -> str:
        rep = ""
        for i in range(9):
            rep += str(list(self.table[i]))[1:-1] + "\n"

        return rep

    def load_sudoku_image(self):

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

    def add(self, number: int, position: tuple) -> None:
        self.table[position] = number

    def is_valid(self, number: int, position: tuple) -> bool:
        # Check row
        for i in range(9):
            if self.table[position[0]][i] == number and position[1] != i:
                return False

        # Check column
        for i in range(9):
            if self.table[i][position[1]] == number and position[0] != i:
                return False

        # Check box
        box_x = position[1] // 3
        box_y = position[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.table[i][j] == number and (i,j) != position:
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
            

def load_digit(i):

    root_path = r"sudoku-public\public\set\00\digits"
    digit_path = "\\" + str(i) + ".png"

    image_file = Image.open(root_path + digit_path)
    image = np.array(image_file)
    image = 255 - image[:, :, 3]

    img = Image.fromarray(image, 'L')
    image_file = ImageOps.grayscale(img)
    img = np.array(image_file)

    inverted_digit = 255 - img
    inverted_digit[inverted_digit > 0] = 255
    return inverted_digit


if __name__ == "__main__":

    # image_path = input()
    # sudoku = Sudoku(image_path)

    # for i in range(1,10):
    #     digit = load_digit(i)
    #     white_pixels = np.count_nonzero(digit)
    #     black_pixels = digit.size - white_pixels
    #     print(i, white_pixels/black_pixels)


    sudoku = Sudoku()
    # sudoku.solve()
    # print(sudoku)