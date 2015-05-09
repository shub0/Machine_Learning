import sys

class StatusBar:
    def __init__(self, total):
        self.unit = int(total / 100)
        self.total = total
        print '[' + '=' * 100 + ']'
        sys.stdout.write('[')
        sys.stdout.flush()

    def update(self, count):
        if count % self.unit == 0:
            sys.stdout.write('=')
            sys.stdout.flush()

    def finish(self):
        sys.stdout.write(']\n')
        sys.stdout.flush()
