from piano_dataset.piano_task import PianoTask
from piano_dataset.piano_tasks import PianoTaskManager

PianoTasks = PianoTaskManager()
for subclass in PianoTask.__subclasses__():
    PianoTasks.register_task(subclass)
