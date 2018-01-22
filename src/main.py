from torch.autograd import Variable
from src.style_transfer import run_style_transfer
from src.img import load_img
from src.img import save_img

content_img = Variable(load_img('../images/content.png'))
style_img = Variable(load_img('../images/style.png'))
input_img = Variable(load_img('../images/input.png'))

save_img(run_style_transfer(content_img, style_img, input_img, 300))
