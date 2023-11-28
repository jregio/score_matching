from model import Model
from distribution import Distribution

p1 = Distribution('p1', [1], [0], [1])
p2 = Distribution('p2', [1/3, 2/3], [-2.5, 2.5], [0.5, 0.5])

Model(mode='prob', distribution=p1).run()
Model(mode='logprob', distribution=p1).run()
Model(mode='score', distribution=p1).run()
Model(mode='prob', distribution=p2).run()
Model(mode='logprob', distribution=p2).run()
Model(mode='score', distribution=p2).run()