#jobStore.py
import modal
from jobStatus import JobStatus

class JobStore:
    def __init__(self):
        self.cache = modal.Dict.from_name("jobs", create_if_missing=True)

    def createJob(self, jobId: str):
        self.cache[jobId] = JobStatus.PENDING.value

    def getJobStatus(self, jobId: str):
        return self.cache.get(jobId, "not_found")

    def setJobStatus(self, jobId: str, status: JobStatus):
        self.cache[jobId] = status.value
    
    def isJobReady(self, jobId: str):
        if(self.cache[jobId] == JobStatus.COMPLETED.value):
            return True
        else:
            return False
