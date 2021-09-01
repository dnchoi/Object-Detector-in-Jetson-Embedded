import time 

class subscriber:
    def __init__(self):
        self._addr = None
        self._bbox = []
        self._color = []
        self._text = None
        self._cur_time = time.time()
        
    def _callback(self, data):
        cls_data, total_length = self.split_string_data(data.cls)
        conf_data, total_length = self.split_string_data(data.conf)
        x1s, total_length = self.split_string_data(data.x1)
        y1s, total_length = self.split_string_data(data.y1)
        x2s, total_length = self.split_string_data(data.x2)
        y2s, total_length = self.split_string_data(data.y2)
        reds, total_length = self.split_string_data(data.r)
        greens, total_length = self.split_string_data(data.g)
        blues, total_length = self.split_string_data(data.b)
        self._addr = data.Header.frame_id
        self._reso = data.Header.seq
        self._bbox = self.make_bbox(x1s, y1s, x2s, y2s, total_length)
        self._color = self.make_color(reds, greens, blues, total_length)
        self._text = self.make_cls_text(cls_data, total_length)
        self._time = data.Header.stamp.nsecs
        self._cur_time = time.time()
        
    def split_string_data(self, data):
        used_data = data.split(sep=',')
        length = len(used_data)
        return used_data, length

    def make_bbox(self, x1, y1, x2, y2, l):
        a = []
        for i in range(l-1):
            a.append([int(float(x1[i])), int(float(y1[i])), int(float(x2[i])), int(float(y2[i]))])
        return a

    def make_color(self, r, g, b, l):
        a = []
        for i in range(l-1):
            a.append([int(r[i]), int(g[i]), int(b[i])])
        return a

    def make_cls_text(self, cls, l):
        a = []
        for i in range(l-1):
            a.append(cls[i])
        return a
