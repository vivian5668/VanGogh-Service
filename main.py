#!/bin python3

import json
import os

print(os.environ.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI', "fail"))
AWS_REGION = os.environ.get('AWS_REGION', 'us-west-2')
CONTENT_IMAGE_PATH = os.environ.get("CONTENT_IMAGE_PATH", "s3://lambda-pytorch-test/portrait_content_girl.jpeg")
STYLE_IMAGE_PATH = os.environ.get("STYLE_IMAGE_PATH", "s3://lambda-pytorch-test/Starry_Night_style.jpg")
OUTPUT_IMAGE_PATH = os.environ.get("OUTPUT_IMAGE_PATH", "s3://lambda-pytorch-test/test.jpeg")
RESOLUTION = os.environ.get("RESOLUTION", "100")

def get_bucket_and_object_name(s3location):
    """
    Args:
        s3location (str): s3://bucket_name/object_name
    
    Returns:
        tuple (str, str): (bucket_name, object_name)
    """
    splits = s3location[5:].split('/')
    bucket_name = splits[0]
    file_name = "/".join(splits[1:])
    print("Bucket Name:", bucket_name)
    print("filename:", file_name)
    return bucket_name, file_name


def lambda_handler():

    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim #efficient gradient dscents
    
    from PIL import Image
    
    import torchvision.transforms as transforms #transform PIL images into tensors
    import torchvision.models as models #train or load pre-trained models
    import copy # to deep copy the models
    import io
    import boto3
    
    #you can choose to use GPU if available which will make the calculations faster
    device = torch.device("cpu")
    
    
    # this defines how clear the final output will be
    imsize = int(RESOLUTION)
    
    loader = transforms.Compose([
        transforms.Resize(imsize), # scale imported image
        transforms.ToTensor()  #transforms it into a torch tensor
    ])

    
    basewidth = 500
    def getStringIO(filename):
        s3 = boto3.resource('s3', region_name=AWS_REGION)
        bucket_name, file_object = get_bucket_and_object_name(filename)
        bucket = s3.Bucket(bucket_name)
        object = bucket.Object(file_object)
        file_stream = io.BytesIO()
        object.download_fileobj(file_stream)
        return file_stream
    
    def resize_content_maintain_aspect(basewidth, content_image_path):
        img = Image.open(getStringIO(content_image_path))
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        return img
    
    
    def resize_style_image(width, height, style_image_path):
        img = Image.open(getStringIO(style_image_path))
        img = img.resize((width, height), Image.ANTIALIAS)
        return img
    
    
    def image_loader(image):
        #fake batch dimension required to fit network's input dimensions
        image =loader(image).unsqueeze(0)
        return image.to(device, torch.float)
    
    basewidth = 100
    content_image_path = CONTENT_IMAGE_PATH
    content_img = resize_content_maintain_aspect(basewidth, content_image_path)
    
    
    style_image_path = STYLE_IMAGE_PATH
    style_img = resize_style_image(content_img.size[0], content_img.size[1], style_image_path)
    
    style_img = image_loader(style_img)
    content_img = image_loader(content_img)
    
    #normal image
    unloader = transforms.ToPILImage()
    def imgshow(tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
    
    #Content loss function is defined as the $squared mean error$ between 
    #two sets of feature maps of content image and destination image
    #use torch.nn.MSELoss
    
    class ContentLoss(nn.Module):
        def __init__(self, target):
            super(ContentLoss, self).__init__()
            self.target = target.detach()
            
        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input
    
    #style loss
    def gram_matrix(input):
        a, b, c, d = input.size()
        #a is batch size, b is num of feature maps, (c, d)=dimensions of a f. map (N=c*d)
        
        features = input.view(a* b, c * d) 
        
        G = torch.mm(features, features.t())
        
        #normalize the values of gram matrix by dividing num of elements in each f. map
        return G.div(a * b * c * d)
    
    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input
            
    
    # import a pre-trained neural network. We will use a 19 layer VGG network
    # Caffee VGG19 model
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    
    
    #VGG networks are trained on images with each channel normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. We will use them to normalize the image before sending it into the network.
    
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)
        def forward(self, img):
            return (img - self.mean) / self.std
    
    
    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)
    
        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(device)
    
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []
    
        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)
    
        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    
            model.add_module(name, layer)
    
            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
    
            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
    
        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
    
        model = model[:(i + 1)]
    
        return model, style_losses, content_losses
    
    
    #Next, we select the input image. You can use a copy of the content image or white noise.
    input_img = content_img.clone()
    
    
    def get_input_optimizer(input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer
    
    
    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = get_input_optimizer(input_img)
    
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
    
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)
    
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0
    
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
    
                style_score *= style_weight
                content_score *= content_weight
    
                loss = style_score + content_score
                loss.backward()
    
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
    
                return style_score + content_score
    
            optimizer.step(closure)
    
        # a last correction...
        input_img.data.clamp_(0, 1)
    
        return input_img
    
    
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    
    import numpy as np
    from PIL import Image
    
    im = Image.fromarray(np.uint8(output.cpu().detach().numpy()[0,:] * 255).astype('uint8').T)
    im = im.transpose(Image.ROTATE_270)
    s3 = boto3.client('s3', region_name=AWS_REGION)
    import io
    bio = io.BytesIO()
    im.save(bio,format="jpeg")
    bio.seek(0)
    bucket_name, object_name = get_bucket_and_object_name(OUTPUT_IMAGE_PATH)
    s3.put_object(Bucket=bucket_name, Key=object_name, Body=bio)

    return {
        "statusCode": 200,
        "body": json.dumps('Your process has been done!')
    }
lambda_handler()
