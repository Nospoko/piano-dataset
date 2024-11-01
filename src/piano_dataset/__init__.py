from piano_dataset.piano_task import PianoTask
from piano_dataset.piano_tasks import PianoTaskManager

task_manager = PianoTaskManager()
for subclass in PianoTask.__subclasses__():
    task_manager.register_task(subclass)
