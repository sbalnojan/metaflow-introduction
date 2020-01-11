from metaflow import FlowSpec,step

class HellowFlow(FlowSpec):
    @step
    def start(self):
        print("Kaggle Challenge is starting")
        self.next(self.hello)

    @step
    def hello(self):
        print("Hello to challenge")
        self.next(self.end)

    @step
    def end(self):
        print("All done")

if __name__ == "__main__":
    HellowFlow()