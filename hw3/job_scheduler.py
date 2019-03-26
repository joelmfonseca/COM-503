
class JobScheduler():
    def __init__(self):
        self.job_schedule = []
    
    def add_job(self, job_type, firing_time):
        self.job_schedule.append((job_type, firing_time))
        self.job_schedule.sort(key=lambda t: -t[1])

    def pop(self):
        job_type, firing_time = self.job_schedule[-1]
        self.job_schedule = self.job_schedule[:-1]
        return job_type, firing_time
