import torch



def profiler(model, optim,loss_fn, data, LRscheduler, info):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == 'cuda' else torch.float32
    train_loader,_,_ = data.get_loaders(data)
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(info),
        record_shapes=False, profile_memory= True, with_stack=True)
    prof.start()
    model.train()
    model = model.to(device, dtype = dtype)
    for i, (inputs, labels) in enumerate(train_loader):
        if i >=(1+1+3)*2:
            break
        inputs = inputs.to(device, dtype = dtype)
        labels = labels.to(device, dtype = dtype)
        optim.zero_grad(set_to_none= True)
        outputs = model(inputs).flatten()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()
        prof.step()
    prof.stop()
        # print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
    LRscheduler.step()    