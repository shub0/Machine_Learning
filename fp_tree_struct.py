from collections import defaultdict, namedtuple

class FPNode(object):
    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = dict()
        self._neighbor = None

    def add(self, child):
        if not isinstance(child, FPNode):
            raise TypeError('Can only set FPNodes as children')

        if child.item not in self._children:
            self._children[child.item] = child
            child.parent = self

    def search(self, item):
        return self._children.get(item, None)

    def remove(self, child):
        try:
            if self._children[child.item] is child:
                del self._children[child.item]
                child.parent = None
                self._tree._remove(child)
                for sub_child in child.children:
                    try:
                        self._children[sub_child.item]._count += sub_child.count
                        sub_child.parent = None
                    except KeyError:
                        self.add(sub_child)
                child._children = dict()
            else:
                raise ValueError('Node is not a valid child')
        except KeyError:
            raise ValueError('node is not a valid child')

    def __contains__(self, item):
        return item in self._children

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    @property
    def count(self):
        return self._count

    def increment(self):
        if self._count is None:
            raise ValueError('Root nodes have no associated count')
        self._count += 1

    @property
    def root(self):
        return not self._item and not self._count

    @property
    def leaf(self):
        return len(self._children) == 0

    def parent():
        doc = "The node's parent"
        def fget(self):
            return self._parent
        def fset(self, value):
            if value is not None and not isinstance(value, FPNode):
                raise TypeError('A node must be an FPNode as a parent')
            if value and value.tree is not self.tree:
                raise ValueError('Cannot have a parent from another tree')
            self._parent = value
        return locals()
    parent = property(**parent())

    def neighbor():
        doc = "The node's neighbour"
        def fget(self):
            return self._neighbor
        def fset(self, value):
            if value is not None and not isinstance(value, FPNode):
                raise TypeError('A node must be an FPNode as a neighbor')
            if value and value.tree is not self.tree:
                raise ValueError('Cannot have a neighbor from another tree')
            self._neighbor = value
        return locals()
    neighbor = property(**neighbor())

    @property
    def children(self):
        return tuple(self._children.itervalues())

    def insepct(self, depth=0):
        print (' ' * depth) + repr(self)
        for child in self.children:
            child.inspect(depth + 1)

    def __reprt__(self):
        if self.root:
            return '<%s (root)>' % type(self).__name__
        return '<%s %r (%r)>' % (type(self).__name__, self.item, self.count)

class FPTree(object):
    Route = namedtuple('Route', 'head tail')
    def __init__(self):
        self._root = FPNode(self, None, None)
        self._routes = dict()

    @property
    def root(self):
        return self._root

    def add(self, transaction):
        point = self._root
        for item in transaction:
            next_point = point.search(item)
            if next_point:
                # reuse exsiting nodes
                next_point.increment()
            else:
                next_point = FPNode(self, item)
                point.add(next_point)
                # update the route of nodes that contains this item to include the new node
                self._update_route(next_point)
            point = next_point

    def _update_route(self, point):
        assert self is point.tree
        try:
            route = self._routes[point.item]
            route[1].neighbor = point
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        for item in self._routes.iterkeys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        try:
            node = self._routes[item][0]
        except KeyError:
            return
        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        def collect_path(node):
            path = list()
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path
        return (collect_path(node) for node in self.nodes(item))

    def inspect(self):
        print 'Tree:'
        self.root.inspect(1)
        print '\nRoutes:'
        for item, nodes in self.items():
            print '   %r' % item
            for node in nodes:
                print '    %r' % node

    def _remove(self, node):
        head, tail = self._routes[node.item]
        if node is head:
            if node is tail or not node.neighbor:
                del self._routes[node.item]
            else:
                self._routes[node.item] = self.Route(node.neighbor, tail)
        else:
            for n in self.nodes(node.item):
                if n.neighbor is node:
                    n.neighbor = node.neighbor
                    if node is tail:
                        self._routes[node.item] = self.Route(head, n)
                    break
