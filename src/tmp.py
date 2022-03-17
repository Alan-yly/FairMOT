from lib.mywork import dataset
import yaml
config = list(yaml.safe_load_all(open('config.yaml')))[0]
print(config)
train_config = config['train']
dt = dataset.Dataset(train_config)
dt.__getitem__(2965)