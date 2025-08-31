class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, x):
        """Create a new set containing only element x"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def find(self, x):
        """Find the representative element of the set containing x (with path compression)"""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Merge two sets containing x and y (union by rank)"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
    
    def get_sets(self):
        """Get all disjoint sets"""
        sets = {}
        for x in self.parent:
            root = self.find(x)
            if root not in sets:
                sets[root] = set()
            sets[root].add(x)
        return list(sets.values())
