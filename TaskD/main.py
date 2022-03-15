import numpy as np
from PIL import Image
from math import ceil

MOVEMENTS = [[-1, 0], [0, -1], [0, 1], [1, 0]]
MOV_MAP = {
    (-1, 0): 'u',
    (0, -1): 'l',
    (0, 1): 'r',
    (1, 0): 'd'
}

class RoombaHandler():

    def __init__(self, inp):

        self.image_path = inp[0]
        self.C = inp[1]
        self.Cmax = inp[2]

        self.obj_dict = {
            0: 'empty',
            1: 'wall',
            2: 'roomba',
            3: 'station',
            4: 'furniture',
            5: 'dirt'
        }

        self.parse_image()
        self.parse_tiles()

        print(f"[{self.C}, {self.Cmax}]")
        self.print_task1()

        self.roomba_pos = np.unravel_index(np.argmax(self.room_matrix == 2), shape=self.room_matrix.shape)
        self.charger_pos = np.unravel_index(np.argmax(self.room_matrix == 3), shape=self.room_matrix.shape)

        self.dirts_pos = list(map(lambda x: tuple(x), np.argwhere(self.room_matrix == 5)))

        self.dirt_mapping = {self.dirts_pos[i]: i for i in range(len(self.dirts_pos))}
        self.floyd = np.zeros((len(self.dirts_pos), len(self.dirts_pos)))
        self.floyd2 = np.zeros((len(self.dirts_pos)+1, len(self.dirts_pos)+1))

    def solve_and_print(self):

        path2 = self.bfs(self.roomba_pos, self.charger_pos)

        self.print_task2(path2)

        for i in range(len(self.dirts_pos)):
            for j in range(i+1, len(self.dirts_pos)):
                self.floyd[i,j] = len(self.bfs(self.dirts_pos[i], self.dirts_pos[j]))
        self.floyd += self.floyd.T + np.eye(len(self.dirts_pos))*max(self.num_cols, self.num_rows)
        
        path3 = self.solve_task3()
        self.print_task3(path3)

        self.dirt_pos_and_charger = self.dirts_pos + [self.charger_pos]

        for i in range(len(self.dirt_pos_and_charger) + 1):
            for j in range(i+1, len(self.dirt_pos_and_charger) + 1):
                self.floyd2[i,j] = len(self.bfs(self.dirt_pos_and_charger[i], self.dirt_pos_and_charger[j]))
        self.floyd2 += self.floyd2.T + np.eye(len(self.dirt_pos_and_charger)+1)*1000

        path4 = self.solve_task4()
        self.print_task4(path4)

        # num_chargings, path5 = self.solve_task5()
        # self.print_task5(num_chargings, path5)

    def parse_image(self):

        # Read image
        image = np.array(Image.open(self.image_path))[:,:,:3]

        mean_image = np.mean(image, axis=2).astype(int)
        height, width = mean_image.shape

        cols = np.mean(mean_image, axis=0)
        rows = np.mean(mean_image, axis=1)
        
        upper_left_x = np.argmax(rows < 255)
        upper_left_y = np.argmax(cols < 255)

        lower_right_x = height - 1 - np.argmax(np.flip(rows) < 255)
        lower_right_y = width - 1 - np.argmax(np.flip(cols) < 255)

        # Crop original image
        self.image = image[upper_left_x:lower_right_x, upper_left_y:lower_right_y]

        # Image shape
        self.H, self.W = self.image.shape[:2]

        # Tile shape
        self.Wt = np.argmax(mean_image[upper_left_x, upper_left_y:])
        self.Ht = np.argmax(mean_image[upper_left_x:, upper_left_y])

        # Set number of cols
        self.num_rows = self.H // self.Ht
        self.num_cols = self.W // self.Wt

        self.Hs = ceil((self.H - self.num_rows*self.Ht) / (self.num_rows - 1))
        self.Ws = ceil((self.W - self.num_cols*self.Wt) / (self.num_cols - 1))

    def get_tile(self, row, col):
        return self.image[row*(self.Ht+self.Hs):min(self.H, (row+1)*(self.Ht+self.Hs)-self.Hs), col*(self.Wt+self.Ws): min(self.W, (col+1)*(self.Wt+self.Ws)-self.Ws)]

    def parse_tiles(self):
        def tile_to_obj(tile_image):

            ht, wt = tile_image.shape[:2]

            non_white = tile_image == np.ones((ht, wt, 3))*255
            non_white = ~(non_white[:,:,0] & non_white[:,:,1] & non_white[:,:,2])
            non_white_sum = non_white.sum()
            if non_white_sum > 0:
                Rd = self._descretize(np.median(tile_image[:,:,0][non_white]).astype(int))
                Gd = self._descretize(np.median(tile_image[:,:,1][non_white]).astype(int))
                Bd = self._descretize(np.median(tile_image[:,:,2][non_white]).astype(int))
            else:
                Rd, Gd, Bd = 2, 2, 2

            if (Rd, Gd, Bd) == (0,0,0):
                if non_white_sum / (ht * wt) < 0.1:
                    Rd, Gd, Bd = 2, 2, 2


            return self._color_mapper((Rd, Gd, Bd))
            
        self.room_matrix = np.array([[tile_to_obj(self.get_tile(r, c)) for r in range(self.num_rows)] for c in range(self.num_cols)]).T

    def _descretize(self, x):

        if x < 85:
            return 0
        elif x < 170:
            return 1
        else:
            return 2

    def _color_mapper(self, rgb):

        if rgb == (0,0,0):
            return 1
        elif rgb == (2,0,0):
            return 2
        elif rgb == (0,1,0):
            return 3
        elif rgb == (1,0,0):
            return 4
        elif rgb == (2,2,0):
            return 5
        else:
            return 0

    def print_task1(self):
        print('Task 1')
        print(f"[{self.num_rows}, {self.num_cols}]")
        for i in range(self.room_matrix.shape[0]):
            print("[" + (', ').join(list(map(lambda x: str(x), self.room_matrix[i].tolist()))) + "]")

    def bfs(self, start, target):

        def parse_neighbours(prev, curr):
            global MOV_MAP, MOVEMENTS
            for mov in MOVEMENTS:
                if (prev[0] + mov[0], prev[1] + mov[1]) == curr:
                    return MOV_MAP[tuple(mov)]

        queue = [(start, -1, 0)]
        visited = np.zeros_like(self.room_matrix)
        previous_node_dict = {}

        while len(queue):
            node, prev_node, level = queue.pop(0)
            visited[node[0], node[1]] = 1
            previous_node_dict[node] = prev_node

            if node == target:
                path = []
                while previous_node_dict[node] != -1:
                    path.append(parse_neighbours(previous_node_dict[node], node))
                    node = previous_node_dict[node]
                path.reverse()
                return path

            for mov in MOVEMENTS:
                pos = (node[0]+mov[0], node[1]+mov[1])
                if self.room_matrix[pos[0], pos[1]] not in [1, 4]:
                    if not visited[pos[0], pos[1]]:
                        queue.append((pos, node, level+1))

    def solve_task3(self):

        n = len(self.dirts_pos)
        dp = np.zeros((n, 2 ** n), dtype=int)
        ham_length = np.zeros((n, 2 ** n), dtype=int)
 
        for i in range(n):
            dp[i][2 ** i] = len(self.bfs(self.roomba_pos, self.dirts_pos[i]))
            ham_length[i][2 ** i] = len(self.bfs(self.roomba_pos, self.dirts_pos[i]))
 
        for i in range(2 ** n):
            for j in range(n):
                if (2 ** j) & i:
                    mn = np.inf
                    for k in range(n):
                        if j != k and ((2 ** k) & i):
                            if dp[k][i ^ (2 ** j)]:
 
                                if mn > ham_length[k][i ^ (2 ** j)] + self.floyd[k][j]:
                                    mn = ham_length[k][i ^ (2 ** j)] + self.floyd[k][j]
                                    ham_length[j][i] = ham_length[k][i ^ (2 ** j)] + self.floyd[k][j]
 
                                    dp[j][i] = (k + 1)
 
        min_length = 10000000
        for i in range(n):
            min_length = min(min_length, ham_length[i][(2 ** n) - 1])
 
        for i in range(1, n + 1):
            length = ham_length[i - 1][(2 ** n) - 1]
 
            if length == min_length:
                path = []
                curr = i
                bitmask = (2 ** n) - 1
                while True:
                    prev = dp[curr - 1][bitmask]
 
                    if (2 ** (curr - 1)) == bitmask:
                        temp_path = self.bfs(self.roomba_pos, self.dirts_pos[curr - 1])
                        temp_path.reverse()
 
                        path.extend(temp_path)
                        break
                    else:
                        temp_path = self.bfs(self.dirts_pos[prev - 1], self.dirts_pos[curr - 1])
                        temp_path.reverse()
 
                        path.extend(temp_path)
 
                    bitmask ^= (2 ** (curr - 1))
                    curr = prev
 
                path.reverse()
                break
 
        return path

    def solve_task4(self):

        n = len(self.dirts_pos)
        dp = np.zeros((n, 2 ** n), dtype=int)
        ham_length = np.zeros((n, 2 ** n), dtype=int)

        for i in range(n):
            dp[i][2 ** i] = len(self.bfs(self.roomba_pos, self.dirts_pos[i]))
            ham_length[i][2 ** i] = len(self.bfs(self.roomba_pos, self.dirts_pos[i]))
 
        for i in range(2 ** n):
            for j in range(n):
                if (2 ** j) & i:
                    mn = np.inf
                    for k in range(n):
                        if j != k and ((2 ** k) & i):
                            if dp[k][i ^ (2 ** j)]:
 
                                if mn > ham_length[k][i ^ (2 ** j)] + self.floyd[k][j]:
                                    mn = ham_length[k][i ^ (2 ** j)] + self.floyd[k][j]
                                    ham_length[j][i] = ham_length[k][i ^ (2 ** j)] + self.floyd[k][j]
 
                                    dp[j][i] = (k + 1)
 
        min_length = 10000000
        for i in range(n):
            min_length = min(min_length, ham_length[i][(2 ** n) - 1])
 
        for i in range(1, n + 1):
            length = ham_length[i - 1][(2 ** n) - 1]
 
            if length == min_length:
                path = []
                curr = i
                bitmask = (2 ** n) - 1
                while True:
                    prev = dp[curr - 1][bitmask]
 
                    if (2 ** (curr - 1)) == bitmask:
                        temp_path = self.bfs(self.roomba_pos, self.dirts_pos[curr - 1])
                        temp_path.reverse()
 
                        path.extend(temp_path)
                        break
                    else:
                        temp_path = self.bfs(self.dirts_pos[prev - 1], self.dirts_pos[curr - 1])
                        temp_path.reverse()
 
                        path.extend(temp_path)
 
                    bitmask ^= (2 ** (curr - 1))
                    curr = prev
 
                path.reverse()
                break
 
        return path


    def hamilton_path_finding(self):
        n = len(self.dirts_pos)
        dp = np.zeros((n, 2 ** n), dtype=int)
        
        for i in range(n):
            dp[i][2 ** i] = len(self.bfs(self.roomba_pos, self.dirts_pos[i]))

        for i in range(2 ** n):
            for j in range(n):
                if ((2 ** j) & i):
                    for k in range(n):
                        if (j != k and (2 ** k) & i):
                            if dp[k][i ^ (2 ** j)]:
                                
                                dp[j][i] = (k + 1)
                                break

        min_length = 10000000
        for i in range(1, n + 1):
            curr = i 
            length = 0
            bitmask = (2 ** n) - 1

            while True:
                prev = dp[curr - 1][bitmask]

                if (2 ** (curr - 1)) == bitmask:
                    length += dp[curr - 1][bitmask]
                    break
                else:
                    length += self.floyd[prev - 1][curr - 1]
                
                bitmask ^= (2 ** (curr - 1))
                curr = prev

            
            min_length = min(length, min_length)

        for i in range(1, n + 1):
            curr = i 
            length = 0
            bitmask = (2 ** n) - 1

            while True:
                prev = dp[curr - 1][bitmask]

                if (2 ** (curr - 1)) == bitmask:
                    length += dp[curr - 1][bitmask]
                    break
                else:
                    length += self.floyd[prev - 1][curr - 1]
                
                bitmask ^= (2 ** (curr - 1))
                curr = prev

            
            if length == min_length:
                path = []
                curr = i
                bitmask = (2 ** n) - 1
                while True:
                    prev = dp[curr - 1][bitmask]

                    if (2 ** (curr - 1)) == bitmask:
                        temp_path = self.bfs(self.roomba_pos, self.dirts_pos[curr - 1])
                        temp_path.reverse()
                        path.extend(temp_path)
                        break
                    else:
                        temp_path = self.bfs(self.dirts_pos[prev - 1], self.dirts_pos[curr - 1])
                        temp_path.reverse()
                        path.extend(temp_path)
                    
                    bitmask ^= (2 ** (curr - 1))
                    curr = prev

                path.reverse()
                break

        return path

    def print_task2(self, path):
        print("Task 2")
        print("[\'" + '\', \''.join(path) + "\']")

    def print_task3(self, path):
        print("Task 3")
        print("[\'" + '\', \''.join(path) + "\']")

    def print_task4(self, path):
        print("Task 4")
        print("[\'" + '\', \''.join(path) + "\']")

    def print_task5(self, num_chargings, path):
        print("Task 5")
        print("[" + str(num_chargings) + "\'" + '\', \''.join(path) + "\']")

if __name__ == "__main__":

    # inp = input().split(' ')

    # za path4 koji je jednak path3 -> najbolji

    inps = {3: (r"TaskD\dataset\public\set\room_1.png", 2,6),
    5: (r"TaskD\dataset\public\set\room_2.png", 2,6), 
    7: (r"TaskD\dataset\public\set\room_3.png", 7,12),
    8: (r"TaskD\dataset\public\set\room_3.png", 9,9),
    9: (r"TaskD\dataset\public\set\room_3.png", 9,10),
    10: (r"TaskD\dataset\public\set\room_3.png", 7,8),
    11: (r"TaskD\dataset\public\set\room_4.png", 3,8),
    13: (r"TaskD\dataset\public\set\room_5.png", 3,6),
    14: (r"TaskD\dataset\public\set\room_5.png", 3,7),
    17: (r"TaskD\dataset\public\set\room_7.png", 5,8)
    }

    # za path4 koji je jednak hamilton_path_finding

    # inps = {3: (r"TaskD\dataset\public\set\room_1.png", 2,6),
    # 5: (r"TaskD\dataset\public\set\room_2.png", 2,6), 
    # 7: (r"TaskD\dataset\public\set\room_3.png", 7,12),
    # 8: (r"TaskD\dataset\public\set\room_3.png", 9,9),
    # 9: (r"TaskD\dataset\public\set\room_3.png", 9,10),
    # 10: (r"TaskD\dataset\public\set\room_3.png", 7,8),
    # 12: (r"TaskD\dataset\public\set\room_4.png", 9,7), # wtf???
    # 13: (r"TaskD\dataset\public\set\room_5.png", 3,6),
    # 14: (r"TaskD\dataset\public\set\room_5.png", 3,7),
    # 17: (r"TaskD\dataset\public\set\room_7.png", 5,8)
    # }

    for k in inps.keys():
        print(f'------------ case {k} ------------')
        roomba = RoombaHandler(inps[k])
        roomba.solve_and_print()
        del roomba