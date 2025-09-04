#region modules
from typing import List
#endregion

#region variables
#endregion

#region functions
#endregion

#region classes
class Profiler:
    def __init__(self, args):
        self.args = args
    
    @property
    def file_contents(self) -> dict:
        return {}

    def write(self):
        for filename, filecontent in self.file_contents.items():
            with open(filename, 'w') as f:
                f.write(filecontent)
                print(f'Done writing {filename}.', flush=True)

    def plot(self):
        pass

#endregion