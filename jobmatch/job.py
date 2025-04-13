class JobTitleClassificationPath:
    path: [str]
    "The path from the root of taxonomy hierarchy to the job title"

    def __init__(self, path):
        self.path = path

    def get_title(self):
        return self.path[-1]

    def __str__(self):
        return ' > '.join(self.path)
