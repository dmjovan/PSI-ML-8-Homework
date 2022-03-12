from PIL import Image, ImageOps
import numpy as np

class SudokuTable:

    def __init__(self) -> None:
        # self.table = np.zeros((9,9), dtype=int)
        self.table = np.array([
                        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7]
                    ], dtype=int)

    def __repr__(self) -> str:
        rep = ""
        for i in range(9):
            rep += str(list(self.table[i]))[1:-1] + "\n"

        return rep

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
            

if __name__ == "__main__":

    # path = input()
    digits_path = r"sudoku-public\public\set\00\digits"
    image_path = r"sudoku-public\public\set\00\00.png"

    image_file = Image.open(image_path)
    image_file = ImageOps.grayscale(image_file)
    image = np.array(image_file)

    # print(image.shape)

    sudoku = SudokuTable()
    sudoku.solve()
    print(sudoku)