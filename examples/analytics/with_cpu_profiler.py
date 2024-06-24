import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity



model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

## Should have an option to select mode (test with profiler or not).
## If the selected option is not, then we uses model(inputs) only.
## Use GPU or CPU
with profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True, record_shapes=True) as prof:
    ## With 
    with record_function("model_inference"):
        model(inputs)


# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
# prof.export_chrome_trace("trace.json")
prof.export_memory_timeline("memory.html")