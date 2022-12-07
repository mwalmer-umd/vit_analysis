# simple progress tracker
import time

class SimpleProgress():
    def __init__(self, start=0, end=1000, step=1, freq=100):
        self.cur = start
        self.end = end
        self.step = step
        self.freq = freq
        self.t_gather = 0.0
        self.t_count = 0
        self.t_start = time.time()
        self.t_last = self.t_start
        print('[%i/%i] - est remaining: X ----------'%(self.cur, self.end), end='\r', flush=True)

    def convert_time(self, t):
        t_mark = 'seconds'
        if t > 3600.0:
            t_mark = 'hours'
            t = t / 3600.0
        elif t > 60.0:
            t_mark = 'minutes'
            t = t / 60.0
        return t, t_mark
        
    def update(self):
        self.cur += self.step
        if self.cur % self.freq == 0:
            t_now = time.time()
            dt = t_now - self.t_last
            self.t_gather += dt
            self.t_count += 1
            step_est = self.t_gather / self.t_count
            step_rem = (self.end - self.cur) / self.freq
            t_est = step_rem * step_est
            t_est, t_mark = self.convert_time(t_est)
            print('[%i/%i] - est remaining: %.2f %s ----------'%(self.cur, self.end, t_est, t_mark), end='\r', flush=True)
            self.t_last = t_now

    def finish(self):
        print('')
        t_tot = time.time() - self.t_start
        t_tot, t_mark = self.convert_time(t_tot)
        print('Finished in %.2f %s'%(t_tot, t_mark))