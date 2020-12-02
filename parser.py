import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.9,
                    help='decay factor of reward(default: 0.9)')
parser.add_argument('--tau', type=float, default=1.0,
                    help= 'parameter for GAE computation(default: 1.0)')
parser.add_argument('--time_step', type=int, default=50,
                    help='number of training step for each update(default: 50)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--workers', type=int, default=8,
                    help='number of workers to be used(default: 8)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--critic-loss-coef', type=float, default=0.5,
                    help='critic loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-episodes', type=int, default=100000,
                    help='maximum length of an episode (default: 100000)')
parser.add_argument('--num-non-sample', type=int,default=0,
                    help='number of non sampling processes (default: 2)')
parser.add_argument('--seed', type=int, default=6,
                    help='random seed (default: 6)')
parser.add_argument('--save-interval', type=int, default=10,
                    help='model save interval (default: 10)')
parser.add_argument('--game', type=str, default='SonicTheHedgehog-Genesis',
                    help='name of game')
parser.add_argument('--state', type=str, default='GreenHillZone.Act3',
                    help='chosen state of the game')
