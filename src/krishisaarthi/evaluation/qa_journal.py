"""Durable append-only slot consumption; an uncertain reservation is never reusable."""
import json
import os
from pathlib import Path


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_json(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);fsync_directory(path.parent)


class AttemptJournal:
    def __init__(self,directory,slots):
        self.directory=Path(directory);self.slots=set(slots)
        self.directory.mkdir(mode=0o700,exist_ok=True)

    def before(self,case_id,arm):
        key=(case_id,arm)
        if key not in self.slots:raise ValueError('unplanned_attempt')
        # O_EXCL is the no-retry boundary. Existing started/completed/uncertain all consume the slot.
        path=self.directory/(case_id+'--'+arm+'.json')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:
            json.dump({'case_id':case_id,'arm':arm,'state':'reserved_consumed','attempt_consumed':True},f)
            f.flush();os.fsync(f.fileno())
        fsync_directory(self.directory)

    def after(self,row):
        path=self.directory/(row['case_id']+'--'+row['arm']+'.json')
        if not path.is_file():raise ValueError('missing_attempt_reservation')
        durable_json(path,{'state':'completed_record','attempt_consumed':True,'record':row})
