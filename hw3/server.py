import numpy as np
import collections

from job_scheduler import JobScheduler

class Server():
    def __init__(self, max_req, arrival_rate):
        self.max_req = max_req
        self.arrival_rate = arrival_rate

        self.buffer_t1 = []
        self.buffer_t2 = []

        self.job_schedule = JobScheduler()

        self.history_num_job_arrived = []
        self.history_num_job_served = []

    def gen_service_duration_t1(self):
        return np.random.lognormal(mean=1.5, sigma=0.6)

    def gen_service_duration_t2(self):
        return np.random.uniform(low=0.6, high=1)

    def update_buffer(self, time, buffer, mode):
        if mode == 'add':
            buffer.append((time, +1))
        elif mode == 'delete':
            buffer.append((time, -1))

    def process_buffers(self):

        def process_buffer(buffer):

            # merge values with same key
            buffer_dict = {}
            for time, e in buffer:
                if time in buffer_dict.keys():
                    buffer_dict[time].append(e)
                else:
                    buffer_dict[time] = [e]
            
            # print(buffer_dict)
            for key, value in buffer_dict.items():
                buffer_dict[key] = sum(value)
            l = list(buffer_dict.items())

            # sort entries
            l.sort(key=lambda t: t[0])
            # print(l)
            # convert value to cumulative sum
            cum_sum = 0
            for i in range(len(l)):
                cum_sum += l[i][1]
                l[i] = (l[i][0], cum_sum)

            return l

        self.buffer_t1 = process_buffer(self.buffer_t1)
        self.buffer_t2 = process_buffer(self.buffer_t2)

    def run(self):
        num_job_arrived = 0
        num_job_served = 0
        arrival_time = 0
        last_end_waiting_time_t1 = 0
        last_end_waiting_time_t2 = 0
        while num_job_arrived < self.max_req or self.job_schedule.is_not_empty():

            if num_job_arrived < self.max_req:
                # create new job
                arrival_time = arrival_time + np.random.exponential(scale=1/self.arrival_rate)
                # print('arrival time: {:.3f} \nlast end waiting time: {:.3f} \n '.format(arrival_time, last_end_waiting_time_t1))

                # if arrival_time < last_end_waiting_time_t1:
                #     print('it is possible') # never happens

                end_waiting_time_t1 = max(arrival_time, last_end_waiting_time_t1)
                last_end_waiting_time_t1 = end_waiting_time_t1

                self.job_schedule.add_job(1, end_waiting_time_t1)
                self.update_buffer(arrival_time, self.buffer_t1, 'add')
                num_job_arrived += 1
                self.history_num_job_arrived.append((arrival_time, num_job_arrived))

            # pop job from scheduler
            curr_type_job, curr_time = self.job_schedule.next()
            # print('current type job: {}\ncurrent_time: {:.3f}'.format(curr_type_job, curr_time))
            
            if curr_type_job == 1:
                # print('*1')
                self.update_buffer(curr_time, self.buffer_t1, 'delete')
                end_service_time_t1 = curr_time + self.gen_service_duration_t1()
                self.update_buffer(end_service_time_t1, self.buffer_t2, 'add')
                # print('end service time t1: {:.3f}\nlast_end_waiting_time_t2: {:.3f}'.format(end_service_time_t1, last_end_waiting_time_t2))

                end_waiting_time_t2 = max(end_service_time_t1, last_end_waiting_time_t2)
                self.job_schedule.add_job(2, end_waiting_time_t2)

                last_end_waiting_time_t2 = end_waiting_time_t2

            elif curr_type_job == 2:
                # print('*2')
                self.update_buffer(curr_time, self.buffer_t2, 'delete')
                end_service_time_t2 = curr_time + self.gen_service_duration_t2()
                # print('end service_time_t2: {:.3f}'.format(end_service_time_t2))
                num_job_served += 1
                self.history_num_job_served.append((end_service_time_t2, num_job_served))
            # print('-------')
        self.process_buffers()

        return self.history_num_job_arrived, self.history_num_job_served, \
            self.buffer_t1, self.buffer_t2

