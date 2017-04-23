import random

class Board:
    def __init__(self, colors):
        # self._all_colors = set(colors)
        self._nodes = colors
        self._adj = None

    def add_adj(self, adj_table):
        assert len(adj_table) == len(self._nodes)
        self._adj = adj_table

    def copy(self):
        b = Board(self._nodes.copy())
        b.add_adj(self._adj)
        return b

    def flip(self, i, c):
        if self._nodes[i] == c:
            return False
        for j in self._adj[i]:
            if self._nodes[i] == self._nodes[j]:
                self._nodes[j] = c
        self._nodes[i] = c

    def check(self):
        c = self._nodes[0]
        for x in self._nodes:
            if c != x:
                return False
        return True

    def valid_move(self, i):
        return set([self._nodes[x] for x in self._adj[i]])

    def __len__(self):
        return len(self._nodes)

memo = []
def rec(b, depth):
    assert isinstance(b, Board)
    if depth > 4:
        return False
    if depth == 4:
        if b.check():
            print("solved")
            print(memo)
            exit()
        return
    n = len(b)
    for i in range(n):
        for c in b.valid_move(i):
            _b = b.copy()
            _b.flip(i, c)
            memo.append((i, c))
            rec(_b, depth+1)
            memo.pop(-1)

def main():
    colors = ['b', 'y', 'y', 'p', 'b', 'g', 'g']
    b = Board(colors)
    b.add_adj([
        [1, 2, 3],
        [0, 4],
        [0, 4],
        [0, 4],
        [1, 2, 3, 5, 6],
        [4],
        [4]
    ])
    rec(b, 0)


if __name__ == '__main__':
    main()


#
# class Node:
#
#     def __init__(self, color, adjacent, id):
#         self.c = color
#         self.a = set(adjacent)
#         self.id = id
#
#     def add_adj(self, node):
#         self.a.add(node)
#         node.a.add(self)
#
#     def add_adjs(self, nodes):
#         for x in nodes:
#             self.add_adj(x)
#
#     def turn(self, color):
#         # assert self.c != color
#         if self.c == color:
#             return
#         for x in self.a:
#             if x.c == self.c:
#                 x.c = color
#         self.c = color
#
#     def copy(self):
#         return Node(self.c, self.a.copy(), self.id)
#
#     def __repr__(self):
#         return self.__str__()
#
#     def __str__(self):
#         return str(self.c) + "-> " + ''.join([str(x.id) for x in self.a])
#
# def check(nodes):
#     c = nodes[0].c
#     for x in nodes:
#         if x.a != c:
#             return False
#     return True
#
# memo = []
# colors = ['b', 'y', 'y', 'p', 'b', 'g', 'g']
# all_colors = set(colors)
#
#
# def rec(nodes, depth):
#     if depth > 4:
#         return False
#     if depth == 4 and check(nodes):
#         print("Solved")
#     for i in range(len(nodes)):
#         for c in all_colors:
#             _nodes = [n.copy() for n in nodes]
#             _nodes[i].turn(c)
#             rec(_nodes, depth+1)
#
#
# def main():
#     nodes = [Node(v, [], i) for i, v in enumerate(colors)]
#     nodes[0].add_adjs([nodes[1], nodes[2], nodes[3]])
#     nodes[4].add_adjs([nodes[1], nodes[2], nodes[3]])
#     nodes[4].add_adjs([nodes[5], nodes[6]])
#     rec(nodes, 0)
#     # solved = False
#     # while not solved:
#     #
#     #     nodes = [Node(v, [], i) for i, v in enumerate(colors)]
#     #     nodes[0].add_adjs([nodes[1], nodes[2], nodes[3]])
#     #     nodes[4].add_adjs([nodes[1], nodes[2], nodes[3]])
#     #     nodes[4].add_adjs([nodes[5], nodes[6]])
#     #
#     #     memo = []
#     #
#     #     for step in range(4):
#     #
#     #         i = random.randint(0, len(colors)-1)
#     #         to = nodes[i].c
#     #         while to == nodes[i].c:
#     #             to = random.sample(all_colors, 1)[0]
#     #         memo.append((i, nodes[i].c, to))
#     #         nodes[i].turn(to)
#     #
#     #
#     #     if check(nodes):
#     #         print(memo)
#
#
# if __name__ == '__main__':
#     main()