from torch.autograd import Variable
from style_transfer import run_style_transfer
from load_img import show_img
from load_img import load_img

content_img = Variable(load_img("image/content.png"))
style_img = Variable(load_img("image/style.png"))
input_img = Variable(load_img("image/input.png"))

show_img(run_style_transfer(content_img, style_img, input_img, 300))