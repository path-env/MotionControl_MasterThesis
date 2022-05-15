from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fake_data')
writer.add_image('imagessss', img_grid)
writer.add_graph(model)
writer.add_scalar()