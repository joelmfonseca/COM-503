import numpy as np

from job_scheduler import JobScheduler

class Server():
    def __init__(self, max_req, arrival_rate):
        self.max_req = max_req
        self.arrival_rate = arrival_rate

        self.buffer_t1 = []
        self.buffer_t2 = []

        self.job_schedule = JobScheduler()

        self.history_num_job_arrived = []
        self.history_num_job_executed = []

    def gen_service_time_t1(self):
        return np.random.lognormal(mean=1.5, sigma=0.6)

    def gen_service_time_t2(self):
        return np.random.uniform(low=0.6, high=1)

    def update_buffer(self, time, buffer, mode):
        try:
            last_size = buffer[-1][1]
        except IndexError:
            last_size = 0
        if mode == 'add':
            buffer.append((time, last_size+1))
        elif mode == 'delete':
            buffer.append((time, last_size-1))

    def run(self):
        num_job_arrived = 0
        num_job_served = 0
        arrival_time = 0
        last_waiting_time_t1 = 0
        last_waiting_time_t2 = 0
        for _ in range(self.max_req):

            # create new job
            arrival_time = arrival_time + np.random.exponential(scale=1/self.arrival_rate)
            waiting_time = max(arrival_time, last_waiting_time_t1)
            last_waiting_time_t1 = waiting_time

            self.job_schedule.add_job(1, waiting_time)
            self.update_buffer(arrival_time, self.buffer_t1, 'add')
            num_job_arrived += 1
            self.history_num_job_arrived.append((arrival_time, num_job_arrived))

            # pop job from scheduler
            curr_type_job, curr_time = self.job_schedule.pop()

            if curr_type_job == 1:

                self.update_buffer(curr_time, self.buffer_t1, 'delete')
                waiting_time = max(curr_time, last_waiting_time_t2)
                last_waiting_time_t2 = waiting_time + self.gen_service_time_t1()

                self.job_schedule.add_job(2, last_waiting_time_t2)
                self.update_buffer(curr_time + self.gen_service_time_t1(), self.buffer_t2, 'add')

            elif curr_type_job == 2:

                self.update_buffer(curr_time, self.buffer_t2, 'delete')
                num_job_served += 1
                self.history_num_job_executed.append((curr_time + self.gen_service_time_t1(), num_job_served))
        
        return self.history_num_job_arrived, self.history_num_job_executed, \
            self.buffer_t1, self.buffer_t2

