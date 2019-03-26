import numpy as np
import collections

import pprint

from job_scheduler import JobScheduler

def debug(string):
    if True:
        pprint.pprint(string)

class Server():
    def __init__(self, max_req, arrival_rate):
        self.max_req = max_req
        self.arrival_rate = arrival_rate

        self.buffer = []

        self.job_scheduler = JobScheduler()

        self.history_num_job_arrived = []
        self.history_num_job_served = []

    def gen_service_duration_t1(self):
        return np.random.lognormal(mean=1.5, sigma=0.6)

    def gen_service_duration_t2(self):
        return np.random.uniform(low=0.6, high=1)

    def update_buffer(self, job_type, time, mode):
        if mode == 'add':
            self.buffer.append((job_type, time, +1))
        elif mode == 'delete':
            self.buffer.append((job_type, time, -1))

    def process_buffer(self):

        def process_buffer_single_job(buffer_single_job):

            # # merge values with same key
            # buffer_dict = {}
            # for time, e in buffer_single_job:
            #     if time in buffer_dict.keys():
            #         buffer_dict[time].append(e)
            #     else:
            #         buffer_dict[time] = [e]
            
            # for key, value in buffer_dict.items():
            #     buffer_dict[key] = sum(value) 
            # l = list(buffer_dict.items())

            l = buffer_single_job
            # sort entries
            l.sort(key=lambda t: t[0])

            # convert value to cumulative sum
            cum_sum = 0
            for i in range(len(l)):
                cum_sum += l[i][1]
                l[i] = (l[i][0], cum_sum)

            return l

        def extract_buffer_single_job(buffer):

            buffer_info_t1 = []
            buffer_info_t2 = []
            for job_type, time, e in buffer:
                if job_type == 1:
                    buffer_info_t1.append((time, e))
                elif job_type == 2:
                    buffer_info_t2.append((time, e))

            return buffer_info_t1, buffer_info_t2

        debug('raw buffer: {}'.format(self.buffer))

        buffer_info_t1, buffer_info_t2 = extract_buffer_single_job(self.buffer)

        debug('extracted buffer t1: {}'.format(buffer_info_t1))

        buffer_info_t1 = process_buffer_single_job(buffer_info_t1)
        buffer_info_t2 = process_buffer_single_job(buffer_info_t2)

        return buffer_info_t1, buffer_info_t2

    def run(self):
        num_job_arrived = 0
        num_job_served = 0
        arrival_time = 0
        last_end_waiting_time = 0
        while num_job_arrived < self.max_req or self.job_scheduler.is_not_empty():

            if num_job_arrived < self.max_req:

                # create new job
                arrival_time = arrival_time + np.random.exponential(scale=1/self.arrival_rate)
                # debug('arrival time: {:.3f}, last end waiting time: {:.3f}' \
                #     .format(arrival_time, last_end_waiting_time))

                # end_waiting_time = max(arrival_time, last_end_waiting_time)
                # last_end_waiting_time = end_waiting_time

                self.job_scheduler.add_job(1, arrival_time)
                self.update_buffer(1, arrival_time, 'add')
                num_job_arrived += 1
                self.history_num_job_arrived.append((arrival_time, num_job_arrived))

            # pop job from scheduler
            curr_type_job, curr_time = self.job_scheduler.next()
            # debug('current type job: {}, current_time: {:.3f}'.format(curr_type_job, curr_time))
            
            if curr_type_job == 1:
                # debug('*1')
                self.update_buffer(1, curr_time, 'delete')
                end_service_time_t1 = curr_time + self.gen_service_duration_t1()
                self.update_buffer(2, end_service_time_t1, 'add')

                end_waiting_time = max(end_service_time_t1, last_end_waiting_time)
                self.job_scheduler.add_job(2, end_waiting_time)

                last_end_waiting_time = end_waiting_time
                # debug('end service time t1: {:.3f}, last end waiting time: {:.3f}' \
                #     .format(end_service_time_t1, last_end_waiting_time))

            elif curr_type_job == 2:
                # debug('*2')
                self.update_buffer(2, curr_time, 'delete')
                end_service_time_t2 = curr_time + self.gen_service_duration_t2()
                last_end_waiting_time = max(end_service_time_t2, last_end_waiting_time)
                # debug('end service time t2: {:.3f}, last end waiting time: {:.3f}' \
                #     .format(end_service_time_t2, last_end_waiting_time))
                num_job_served += 1
                self.history_num_job_served.append((end_service_time_t2, num_job_served))
            # debug('-------')
            # debug(self.buffer)
            # debug(self.job_scheduler.job_schedule)
            # debug('=======')

        buffer_info_t1, buffer_info_t2 = self.process_buffer()

        return self.history_num_job_arrived, self.history_num_job_served, \
            buffer_info_t1, buffer_info_t2

