import numpy as np
import collections

import pprint

from job_scheduler import JobScheduler

def debug(string):
    if False:
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

        def process_buffer_single_job(buffer):

            # sort entries
            buffer.sort(key=lambda t: t[0])

            # convert value to cumulative sum
            cum_sum = 0
            for i in range(len(buffer)):
                cum_sum += buffer[i][1]
                buffer[i] = (buffer[i][0], cum_sum)

            return buffer

        def extract_buffer_single_job(buffer):

            # split buffer from job types
            buffer_info_t1 = []
            buffer_info_t2 = []
            for job_type, time, e in buffer:
                if job_type == 1:
                    buffer_info_t1.append((time, e))
                elif job_type == 2:
                    buffer_info_t2.append((time, e))

            return buffer_info_t1, buffer_info_t2

        buffer_info_t1, buffer_info_t2 = extract_buffer_single_job(self.buffer)

        buffer_info_t1 = process_buffer_single_job(buffer_info_t1)
        buffer_info_t2 = process_buffer_single_job(buffer_info_t2)

        return buffer_info_t1, buffer_info_t2

    def run(self):
        num_job_arrived = 0
        num_job_served = 0
        arrival_time = 0
        last_end_service_time = 0
        response_time_t1 = []
        response_time_t2 = []
        while num_job_arrived < self.max_req or self.job_scheduler.is_not_empty():

            if num_job_arrived < self.max_req:

                # create new job
                arrival_time = arrival_time + np.random.exponential(scale=1/self.arrival_rate)

                # add new job to job scheduler and buffer
                self.job_scheduler.add_job(1, arrival_time)
                self.update_buffer(1, arrival_time, 'add')

                # keep track of the number of jobs arrived
                num_job_arrived += 1
                self.history_num_job_arrived.append((arrival_time, num_job_arrived))

            # get the next job from the job scheduler
            curr_type_job, service_time = self.job_scheduler.next()
            
            # debug
            debug('arrival time: {:.3f}, current type job: {}, service_time: {:.3f}'\
                .format(arrival_time, curr_type_job, service_time))
            
            if curr_type_job == 1:

                # retrieve arrival time
                arrival_time_t1 = service_time

                # make sure there is no other job going on
                service_time = max(service_time, last_end_service_time)
                self.update_buffer(1, service_time, 'delete')
                
                # compute end of service time for type 1 job
                end_service_time_t1 = service_time + self.gen_service_duration_t1()
                
                # update buffer & job scheduler
                self.update_buffer(2, end_service_time_t1, 'add')
                self.job_scheduler.add_job(2, end_service_time_t1)

                # debug
                debug('1) end service time t1: {:.3f}, last end service time: {:.3f}' \
                    .format(end_service_time_t1, last_end_service_time))

                # update
                last_end_service_time = end_service_time_t1

                # add corresponding response time
                response_time_t1.append(end_service_time_t1-arrival_time_t1)

            elif curr_type_job == 2:

                # retrieve arrival time
                arrival_time_t2 = service_time

                # make sure there is no other job going on
                service_time = max(service_time, last_end_service_time)
                self.update_buffer(2, service_time, 'delete')

                # compute end of service time for type 2 job
                end_service_time_t2 = service_time + self.gen_service_duration_t2()

                # debug
                debug('2) end service time t2: {:.3f}, last end service time: {:.3f}' \
                    .format(end_service_time_t2, last_end_service_time))

                # update
                last_end_service_time = end_service_time_t2

                # keep track of the number of jobs served
                num_job_served += 1
                self.history_num_job_served.append((end_service_time_t2, num_job_served))
            
                # add correspoding response time
                response_time_t2.append(end_service_time_t2-arrival_time_t2)
        
            # debug
            debug('-------')
            debug(self.buffer)
            debug(self.job_scheduler.job_schedule)
            debug('=======')

        buffer_info_t1, buffer_info_t2 = self.process_buffer()

        return self.history_num_job_arrived, self.history_num_job_served, \
            buffer_info_t1, buffer_info_t2, np.mean(response_time_t1), np.mean(response_time_t2)

